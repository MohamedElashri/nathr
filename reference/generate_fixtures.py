"""Generate model-free FP64 examples and a local eager PyTorch checkpoint trace."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np
from safetensors.torch import save_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import check


IDS = [1, 2, 3, 42, 100, 256, 49151, 0]
TEXTS = [
    "", "Hello, world!", "مرحبا بالعالم ١٢٣", "你好，世界！", "café cafe\u0301",
    "  leading\tspaces\r\ntrailing  ", "𐍈🧑🏽‍💻🙂", "a<|endoftext|><file_sep>b",
    "  <repo_name>x <empty_output>  ", "I'm we're can't I'M", "50² Ⅳ ４２ १२",
    "def f(x):\n\treturn x + 42\n", "a" * 129,
]
VERSIONS = ("torch", "transformers", "tokenizers", "safetensors", "numpy")


def tokenizer_examples(tokenizer):
    backend = tokenizer.backend_tokenizer
    encode = backend.encode
    examples = []
    for source in TEXTS:
        raw = encode(source, add_special_tokens=False)
        added = encode(source, add_special_tokens=True)
        examples.append({
            "text": source,
            "pretokenized_before_added_token_matching": backend.pre_tokenizer.pre_tokenize_str(source),
            "offsets": raw.offsets, "tokens": raw.tokens, "ids": raw.ids,
            "ids_with_special_insertion": added.ids,
            "decoded": tokenizer.decode(raw.ids, skip_special_tokens=False),
            "decoded_skip_special": tokenizer.decode(raw.ids, skip_special_tokens=True),
        })

    # These GPT-2 byte alphabet symbols encode individual bytes of U+1F642.
    symbols = ["ð", "Ł", "Ļ", "Ĥ"]
    fragment_ids = [backend.token_to_id(symbol) for symbol in symbols]
    return {
        "versions": {name: metadata.version(name) for name in ("transformers", "tokenizers")},
        "examples": examples,
        "utf8_fragments": {
            "ids": fragment_ids, "raw_bytes_hex": "f09f9982",
            "decoded_prefixes": [tokenizer.decode(fragment_ids[:i]) for i in range(1, 5)],
        },
    }


def fp64_examples():
    x = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    y = np.array([-3.0, 2.0, 4.0], dtype=np.float64)
    eps = 1e-5
    exps = np.exp(x - x.max())
    keys = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, -3.0]])
    values = np.array([[1.0, 2.0], [4.0, -2.0]])
    scores = keys @ x / np.sqrt(x.size)
    probs = np.exp(scores - scores.max())
    return {
        "dtype": "float64", "numpy": metadata.version("numpy"), "x": x.tolist(), "y": y.tolist(),
        "dot": float(x @ y), "rmsnorm_epsilon": eps,
        "rmsnorm": (x / np.sqrt(np.mean(x * x) + eps) * y).tolist(),
        "softmax": (exps / exps.sum()).tolist(),
        "attention": {
            "query": x.tolist(), "keys": keys.tolist(), "values": values.tolist(),
            "output": ((probs / probs.sum()) @ values).tolist(),
        },
    }


def load_hf(path):
    torch.set_num_threads(1)
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()


@torch.inference_mode()
def trace(model):
    outputs = {}
    axes = {}

    def store(name, tensor, order):
        outputs[name] = tensor.detach().contiguous().clone()
        axes[name] = order

    def hook(name, order):
        def capture(_module, _args, output):
            store(name, output, order)
        return capture

    modules = {"embedding": model.model.embed_tokens, "final_norm": model.model.norm}
    layer_axes = ["batch", "token", "hidden"]
    for i, layer in enumerate(model.model.layers):
        for op, module in {
            "input_norm": layer.input_layernorm, "post_attention_norm": layer.post_attention_layernorm,
            "q_before_rope": layer.self_attn.q_proj, "k_before_rope": layer.self_attn.k_proj,
            "value": layer.self_attn.v_proj, "attention_projection": layer.self_attn.o_proj,
            "gate": layer.mlp.gate_proj, "up": layer.mlp.up_proj, "down": layer.mlp.down_proj,
        }.items():
            modules[f"layer.{i}.{op}"] = module
    handles = [module.register_forward_hook(hook(name, ["batch", "token", "feature"]))
               for name, module in modules.items()]
    try:
        result = model(torch.tensor([IDS]), use_cache=True, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()
    store("logits", result.logits, ["batch", "token", "vocabulary"])
    for i, hidden in enumerate(result.hidden_states):
        store(f"hidden_states.{i}", hidden, layer_axes)
    for i, layer in enumerate(result.past_key_values.layers):
        for key, data in (("key_after_rope", layer.keys), ("value", layer.values)):
            store(f"cache.{i}.{key}", data, ["batch", "kv_head", "position", "head_dim"])
    return outputs, axes


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source = next(source for source in manifest["sources"] if source["repo"] == manifest["canonical_model"])
    for item in source["files"]:
        if item["required"]:
            check(args.model / item["filename"], item)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    write_json(args.output / "tokenizer.json", tokenizer_examples(tokenizer))
    write_json(args.output / "fp64.json", fp64_examples())

    model = load_hf(args.model)
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError("native output and embedding weights must alias")
    outputs, axes = trace(model)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in outputs.values()):
        raise ValueError("non-finite or non-FP32 reference tensor")
    save_file(outputs, args.output / "hf.safetensors")
    with (args.output / "hf.safetensors").open("rb") as stream:
        fixture_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json(args.output / "hf.json", {
        "schema_version": 1, "model_revision": source["revision"],
        "model_sha256": next(item["sha256"] for item in source["files"] if item["kind"] == "model"),
        "fixture_sha256": fixture_hash, "versions": {name: metadata.version(name) for name in VERSIONS},
        "token_ids": IDS, "positions": list(range(len(IDS))), "threads": 1, "seed": None,
        "precision": "FP32 eager CPU with cache", "tolerance_profile": "phase0-provisional",
        "tensors": {name: {"shape": list(value.shape), "axes": axes[name], "dtype": "F32"}
                    for name, value in sorted(outputs.items())},
        "greedy_ids": outputs["logits"].argmax(dim=-1).tolist()[0],
    })


if __name__ == "__main__":
    main()
