"""Generate the small checked-in GQA and tokenizer fixtures with the locked reference."""

import argparse
import json
from pathlib import Path

from safetensors.torch import save_file
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


IDS = (1, 7, 2, 18, 0, 3, 21, 13)


def checkpoint(output):
    config = LlamaConfig(
        vocab_size=23, hidden_size=8, intermediate_size=13, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=2,
        max_position_embeddings=8192, bos_token_id=0, eos_token_id=0, tie_word_embeddings=True,
        rope_theta=100000.0, rms_norm_eps=1e-5, attn_implementation="eager",
    )
    config.architectures = ["LlamaForCausalLM"]
    model = LlamaForCausalLM(config).eval()
    generator = torch.Generator().manual_seed(129761)
    weights = {}
    with torch.inference_mode():
        for name, parameter in sorted(model.named_parameters()):
            values = torch.randint(-15, 16, parameter.shape, generator=generator).float() / 64
            if "norm.weight" in name:
                values += 1
            parameter.copy_(values)
            weights[name] = values
        expected = model(torch.tensor([IDS]), use_cache=False).logits[0].contiguous()

    for label, dtype in (("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)):
        folder = output / label
        folder.mkdir(exist_ok=True)
        config.to_json_file(folder / "config.json")
        save_file({name: tensor.to(dtype) for name, tensor in weights.items()}, folder / "model.safetensors")
    save_file({"logits": expected}, output / "expected.safetensors")
    (output / "inputs.json").write_text(json.dumps({"ids": IDS, "seed": 129761, "transformers": "4.57.6"}) + "\n")


def tokenizer(output):
    vocab = {symbol: byte for byte, symbol in bytes_to_unicode().items() if byte != 4}
    # Fill the removed ID with a byte-pair token.
    vocab["ab"] = 4
    vocab["abc"] = 256
    literals = ["<x>", "<x>>"]
    vocab.update((text, 257 + i) for i, text in enumerate(literals))
    stages = {"type": "ByteLevel", "trim_offsets": True, "use_regex": True}
    data = {
        "version": "1.0", "normalizer": None, "post_processor": None, "truncation": None, "padding": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Digits", "individual_digits": True}, {**stages, "add_prefix_space": False}
        ]},
        "decoder": {**stages, "add_prefix_space": True},
        "added_tokens": [
            {"id": vocab[text], "content": text, "special": True, "normalized": False,
             "single_word": False, "lstrip": False, "rstrip": False}
            for text in literals
        ],
        "model": {
            "type": "BPE", "vocab": vocab, "merges": [["a", "b"], ["ab", "c"]],
            "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
            "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
        },
    }
    (output / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    checkpoint(args.output)
    tokenizer(args.output)
