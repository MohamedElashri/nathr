"""Reconstruct an independent FP32 Transformers model from exact GGUF weights."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from gguf import GGUFReader, MODEL_ARCH, dequantize, get_tensor_name_map
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

from generate_fixtures import IDS, load_hf


def reconstruct(file, hf_directory):
    reader = GGUFReader(file)
    config = AutoConfig.from_pretrained(hf_directory, local_files_only=True)
    config._attn_implementation = "eager"
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    config.tie_word_embeddings = "output.weight" not in tensors
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(config).eval()
    names = get_tensor_name_map(MODEL_ARCH.LLAMA, config.num_hidden_layers)
    weights = {}
    storage_types = {}
    for name, parameter in model.state_dict().items():
        gguf_name = names.get_name(name, try_suffixes=(".weight",))
        if name == "lm_head.weight" and config.tie_word_embeddings:
            weights[name] = weights["model.embed_tokens.weight"]
            storage_types[name] = storage_types["model.embed_tokens.weight"]
            continue
        tensor = tensors.pop(gguf_name)
        rows = dequantize(tensor.data, tensor.tensor_type)
        if name.endswith(("q_proj.weight", "k_proj.weight")):
            heads = config.num_key_value_heads if name.endswith("k_proj.weight") else config.num_attention_heads
            # GGUF rows are [head, pair, component], HF rows are [head, component, pair].
            rows = rows.reshape(heads, -1, 2, rows.shape[-1]).transpose(0, 2, 1, 3).reshape(rows.shape)
        if rows.shape != tuple(parameter.shape):
            raise ValueError(f"{file}: unexpected shape for {name}: {rows.shape}")
        weights[name] = torch.from_numpy(rows.astype(np.float32, copy=True))
        if not torch.isfinite(weights[name]).all():
            raise ValueError(f"{file}: non-finite decoded weights for {name}")
        storage_types[name] = tensor.tensor_type.name
    if tensors:
        raise ValueError(f"{file}: unexpected GGUF tensors: {sorted(tensors)}")
    model.load_state_dict(weights, strict=True, assign=True)
    if config.tie_word_embeddings:
        model.tie_weights()
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    if tied != config.tie_word_embeddings:
        raise ValueError(f"{file}: wrong output tying")
    for name, parameter in model.state_dict().items():
        if parameter.data_ptr() != weights[name].data_ptr():
            raise ValueError(f"{file}: installed weight does not share verified storage: {name}")
    return model, storage_types


def exact_floating_check(model, storage_types, file, hf, ids, expected_logits):
    native = hf.state_dict()
    rounding = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
    rounded = None
    changed = 0
    for name, weight in model.state_dict().items():
        expected = native[name].to(rounding[storage_types[name]]).float()
        if not torch.equal(weight, expected):
            raise ValueError(f"{file}: reconstructed floating weight differs at {name}")
        changes = torch.count_nonzero(expected != native[name]).item()
        if changes:
            if rounded is None:
                rounded = deepcopy(hf)
            rounded.state_dict()[name].copy_(expected)
            changed += changes
    baseline = expected_logits if rounded is None else rounded(ids, use_cache=False).logits
    logits = model(ids, use_cache=False).logits
    if not torch.equal(logits, baseline):
        raise ValueError(f"{file}: floating logits mismatch; max abs {(logits - baseline).abs().max().item()}")
    return {
        "weights": "exact_after_storage_rounding", "logits": "exact_vs_equivalently_rounded_hf_fp32",
        "changed_weight_entries": changed,
        "logits_max_abs_vs_unrounded_native": (logits - expected_logits).abs().max().item(),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--published", type=Path)
    parser.add_argument("--controlled", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.published or args.controlled):
        parser.error("specify at least one GGUF suite")
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    entries = []
    if args.controlled:
        entries.extend((args.controlled / item["filename"], item, "controlled")
                       for item in manifest["controlled"]["files"])
    if args.published:
        for source in manifest["sources"]:
            directory = args.published / source["repo"] / source["revision"]
            entries.extend((directory / item["filename"], item, source["repo"])
                           for item in source["files"] if item["filename"].endswith(".gguf"))
    native = load_hf(args.model)
    ids = torch.tensor([IDS])
    expected_logits = native(ids, use_cache=False).logits
    with args.output.open("w") as report:
        for file, item, suite in entries:
            with file.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"{file}: SHA-256 mismatch")
            model, storage_types = reconstruct(file, args.model)
            logits = model(ids, use_cache=False).logits
            if not torch.isfinite(logits).all():
                raise ValueError(f"{file}: non-finite logits")
            result = {
                "suite": suite, "filename": file.name, "sha256": digest, "ids": IDS,
                "greedy_ids": logits.argmax(dim=-1).tolist()[0],
                "output_tied": model.config.tie_word_embeddings,
                "encoding_types": sorted(set(storage_types.values())),
            }
            if suite == "controlled" and item["recipe"] in {"f32", "f16", "bf16"}:
                result.update(exact_floating_check(model, storage_types, file, native, ids, expected_logits))
            report.write(json.dumps(result) + "\n")
            report.flush()
            print(file.name, file=sys.stderr)
            del model


if __name__ == "__main__":
    main()
