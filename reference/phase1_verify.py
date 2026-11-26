"""Validate Phase 1 using the locked HF checkpoint and fast tokenizer oracles."""

import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
from safetensors.numpy import load_file as load_arrays
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama

from generate_fixtures import IDS, load_hf
from models import check


def command(binary, action, model, *options, input_text=None):
    return subprocess.run(
        [binary, action, "--model", model, *options],
        input=input_text, text=True, capture_output=True, check=True,
    ).stdout


@torch.inference_mode()
def capture(model, ids, round_cache=False):
    arrays = {}
    handles = []
    active_layer = None

    def store(name, tensor):
        arrays[name] = tensor.reshape(len(ids), -1).detach().float().numpy().copy()

    def out(name):
        return lambda _module, _args, output: store(name, output)

    def before(name):
        return lambda _module, args: store(name, args[0])

    def select(layer):
        def hook(_module, _args):
            nonlocal active_layer
            active_layer = layer
        return hook

    def rotation(*args, **kwargs):
        query, key = rotate(*args, **kwargs)
        prefix = f"layer.{active_layer}."
        store(prefix + "q_after_rope", query.transpose(1, 2))
        store(prefix + "key_after_rope", key.transpose(1, 2))
        return query, key

    handles.append(model.model.embed_tokens.register_forward_hook(out("embedding")))
    handles.append(model.model.norm.register_forward_hook(out("final_norm")))
    for i, layer in enumerate(model.model.layers):
        prefix = f"layer.{i}."
        attn, mlp = layer.self_attn, layer.mlp
        handles.append(attn.register_forward_pre_hook(select(i)))
        for name, module in {
            "input_norm": layer.input_layernorm, "q_before_rope": attn.q_proj,
            "k_before_rope": attn.k_proj, "value": attn.v_proj, "attention_projection": attn.o_proj,
            "post_attention_norm": layer.post_attention_layernorm,
            "gate": mlp.gate_proj, "up": mlp.up_proj, "down": mlp.down_proj, "layer_output": layer,
        }.items():
            handles.append(module.register_forward_hook(out(prefix + name)))
        for name, module in {
            "layer_input": layer, "attention": attn.o_proj,
            "attention_residual": layer.post_attention_layernorm, "swiglu": mlp.down_proj,
        }.items():
            if name == "layer_input":
                handles.append(module.register_forward_pre_hook(
                    lambda _module, args, kwargs, name=prefix + name: store(
                        name, args[0] if args else kwargs["hidden_states"]), with_kwargs=True))
            else:
                handles.append(module.register_forward_pre_hook(before(prefix + name)))
    rotate = modeling_llama.apply_rotary_pos_emb
    try:
        with patch.object(modeling_llama, "apply_rotary_pos_emb", rotation):
            outputs = model(torch.tensor([ids]), use_cache=True)
            store("logits", outputs.logits)
            for i, layer in enumerate(outputs.past_key_values.layers):
                for name, values in (("key_after_rope", layer.keys), ("value", layer.values)):
                    expected = arrays[f"layer.{i}.{name}"]
                    if round_cache:
                        expected = expected.astype(np.float16).astype(np.float32)
                    np.testing.assert_array_equal(expected, values.transpose(1, 2).reshape(len(ids), -1).numpy())
    finally:
        for handle in handles:
            handle.remove()
    return arrays


def compare(expected, actual, violations=None):
    if expected.keys() != actual.keys():
        raise ValueError(f"trace inventory differs: {expected.keys() ^ actual.keys()}")
    report = {}
    for name, ref in expected.items():
        got = actual[name]
        if got.shape != ref.shape or got.dtype != np.float32 or not np.isfinite(got).all():
            raise ValueError(f"{name}: shape/dtype/non-finite mismatch")
        ref, got = ref.astype(np.float64), got.astype(np.float64)
        error = got - ref
        norm_ref = np.linalg.norm(ref)
        norm_got = np.linalg.norm(got)
        relative_l2 = np.linalg.norm(error) / max(norm_ref, 1e-12)
        cosine = float(np.vdot(ref, got) / (norm_ref * norm_got)) if norm_ref and norm_got else None
        worst_index = list(map(int, np.unravel_index(np.argmax(np.abs(error)), error.shape)))
        max_abs = float(np.max(np.abs(error)))
        if relative_l2 > 1e-4 or (cosine is not None and cosine < 0.99999):
            message = f"{name}: relative L2={relative_l2} cosine={cosine} max={max_abs} at {worst_index}"
            if violations is None:
                raise ValueError(message)
            violations.append(message)
        report[name] = dict(relative_l2=float(relative_l2), cosine=cosine, max_abs=max_abs,
                            rmse=float(np.sqrt(np.mean(error ** 2))), worst_index=worst_index)
    actual_greedy = actual["logits"].argmax(axis=-1)
    expected_greedy = expected["logits"].argmax(axis=-1)
    if violations is None:
        np.testing.assert_array_equal(actual_greedy, expected_greedy)
    elif not np.array_equal(actual_greedy, expected_greedy):
        violations.append(f"greedy IDs {actual_greedy.tolist()} differ from {expected_greedy.tolist()}")
    return report


def float_variants(binary, model, output, baseline):
    stored = load_file(model / "model.safetensors")
    reports = {}
    for name, dtype in (("f32", torch.float32), ("f16", torch.float16)):
        converted = output / name
        converted.mkdir(exist_ok=True)
        shutil.copyfile(model / "config.json", converted / "config.json")
        save_file({key: tensor.to(dtype) for key, tensor in stored.items()},
                  converted / "model.safetensors", metadata={"format": "pt"})
        trace = converted / "trace.safetensors"
        command(binary, "forward", converted, "--tokens", ",".join(map(str, IDS)), "--trace", trace)
        actual = load_arrays(trace)
        reports[name] = compare(capture(load_hf(converted), IDS), actual)
        if name == "f32":
            for key in baseline:
                np.testing.assert_array_equal(actual[key], baseline[key])
    return reports


def check_tokenizer(binary, model, fixtures, oracle_model=None, *, random_cases=2000, seed=4038191, extra=()):
    tokenizer = AutoTokenizer.from_pretrained(oracle_model or model, local_files_only=True, use_fast=True)
    expected = json.loads(fixtures.read_text())
    seeded = random.Random(seed)
    samples = [item["text"] for item in expected["examples"]]
    samples.extend(chr(cp) for cp in range(0x80))
    samples.extend(["ab\x04ab", "\u088fA", "\U00011de0 12", "<|im_start|><empty_output>", "  " * 256 + "word"])
    samples.extend(extra)
    alphabet = " Arab中文جان123١٢٣٤４Ⅳ²\t\n\r" + "\u0301é🙂𐍈 's'!\u00a0\u2000\u2028<|"
    scalars = [chr(point) for point in [seeded.randrange(0x110000) for _ in range(4096)] if not 0xD800 <= point < 0xE000]
    for _ in range(random_cases):
        size = seeded.randrange(1, 96)
        samples.append("".join(seeded.choice(scalars if seeded.randrange(4) == 0 else alphabet) for _ in range(size)))
    output = command(binary, "tokenize", model, "--stdin", "--stages",
                     input_text="".join(json.dumps(text) + "\n" for text in samples))
    outputs = [json.loads(line) for line in output.rstrip("\n").split("\n")]
    for index, (text, actual) in enumerate(zip(samples, outputs, strict=True)):
        encoded = tokenizer(text, add_special_tokens=True).input_ids
        no_special = tokenizer(text, add_special_tokens=False).input_ids
        segments = [value for value, _ in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)]
        reference = {"ids": encoded, "decoded": tokenizer.decode(encoded),
                     "decoded_skip_special": tokenizer.decode(encoded, skip_special_tokens=True),
                     "pretokenized": segments}
        if encoded != no_special or actual != reference:
            raise ValueError(f"tokenizer case {index} {text!r}: expected {reference!r}, got {actual!r}")
    fragments = expected["utf8_fragments"]
    for i, value in enumerate(fragments["decoded_prefixes"], 1):
        ids = ",".join(map(str, fragments["ids"][:i]))
        actual = json.loads(command(binary, "tokenize", model, "--tokens", ids))
        if actual["decoded"] != value or actual["raw_hex"] != fragments["raw_bytes_hex"][:2 * i]:
            raise ValueError(f"UTF-8 fragment mismatch at {i} tokens")
    return {"cases": len(samples), "random_seed": seed, "fragment_prefixes": 4}


@torch.inference_mode()
def check_medium_and_generation(binary, directory, model):
    ids = [17 + (i * 7919 + 12345) % (49152 - 17) for i in range(64)]
    reference = model(torch.tensor([ids]), use_cache=False, logits_to_keep=1).logits[0].numpy()
    logits = {}
    for name, flags in (("cached", ["--cached", "--context", "8192"]), ("uncached", [])):
        result = json.loads(command(binary, "forward", directory, "--tokens", ",".join(map(str, ids)), "--json", *flags))
        logits[name] = np.array(result["logits"], dtype=np.float32)[None]
    np.testing.assert_array_equal(logits["cached"], logits["uncached"])
    metrics = compare({"logits": reference}, {"logits": logits["cached"]})["logits"]

    prompt = "Once upon a time"
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=True)
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    expected = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                              max_new_tokens=8, pad_token_id=0)[0, ids.size(1):].tolist()
    a = json.loads(command(binary, "generate", directory, "--prompt", prompt, "--greedy", "--json", "--max-new-tokens", "8"))
    b = json.loads(command(binary, "generate", directory, "--tokens", ",".join(map(str, ids[0].tolist())),
                           "--greedy", "--json", "--max-new-tokens", "8"))
    if a["generated_ids"] != expected or b["generated_ids"] != expected or a["consumed"] != len(expected) - 1:
        raise ValueError("generation or emitted/consumed token count differs from the oracle")
    if a["text"] != tokenizer.decode(expected, skip_special_tokens=True) or a["pending_id"] != expected[-1]:
        raise ValueError("generation text or pending token differs from the oracle")
    return {"prompt_tokens": 64, "capacity": 8192, "last_logits": metrics,
            "generation_prompt": prompt, "generated_ids": expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floating-variants", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source = next(source for source in manifest["sources"] if source["repo"] == manifest["canonical_model"])
    for item in source["files"]:
        if item["required"]:
            check(args.model / item["filename"], item)
    args.output.mkdir(parents=True, exist_ok=True)
    tokens = ",".join(map(str, IDS))
    native = load_hf(args.model)
    expected = capture(native, IDS)
    reports = {}
    for mode, options in (("uncached", []), ("cached", ["--cached", "--context", "512"])):
        trace = args.output / f"{mode}.safetensors"
        command(args.binary, "forward", args.model, "--tokens", tokens, "--trace", trace, *options)
        actual = load_arrays(trace)
        reports[mode] = compare(expected, actual)
        if mode == "uncached":
            baseline = actual
        else:
            for key in baseline:
                np.testing.assert_array_equal(actual[key], baseline[key])
    reports["tokenizer"] = check_tokenizer(args.binary, args.model, Path("fixtures/phase0/tokenizer.json"))
    reports["medium_and_generation"] = check_medium_and_generation(args.binary, args.model, native)
    if args.floating_variants:
        reports["converted"] = float_variants(args.binary, args.model, args.output, baseline)
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"{len(expected)} real-model operations per path and {reports['tokenizer']['cases']} tokenizer cases passed")


if __name__ == "__main__":
    main()
