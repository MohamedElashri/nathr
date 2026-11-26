"""Generate a tiny GGUF model and valid Q4_0/Q8_0 blocks using the locked GGUF oracle."""

import argparse
import json
from pathlib import Path
import random

from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFWriter, MODEL_ARCH, dequantize, get_tensor_name_map
import numpy as np
from safetensors.numpy import load_file


def model(output):
    tiny = Path("fixtures/phase1/f32")
    cfg = json.loads((tiny / "config.json").read_text())
    writer = GGUFWriter(output / "tiny.f32.gguf", "llama")
    writer.add_vocab_size(cfg["vocab_size"])
    writer.add_context_length(cfg["max_position_embeddings"])
    writer.add_embedding_length(cfg["hidden_size"])
    writer.add_feed_forward_length(cfg["intermediate_size"])
    writer.add_block_count(cfg["num_hidden_layers"])
    writer.add_head_count(cfg["num_attention_heads"])
    writer.add_head_count_kv(cfg["num_key_value_heads"])
    writer.add_rope_dimension_count(cfg["head_dim"])
    writer.add_rope_freq_base(cfg["rope_theta"])
    writer.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    writer.add_bos_token_id(cfg["bos_token_id"])
    writer.add_eos_token_id(cfg["eos_token_id"])
    names = get_tensor_name_map(MODEL_ARCH.LLAMA, cfg["num_hidden_layers"])
    for name, weights in sorted(load_file(tiny / "model.safetensors").items()):
        if name.endswith(("q_proj.weight", "k_proj.weight")):
            heads = cfg["num_key_value_heads"] if name.endswith("k_proj.weight") else cfg["num_attention_heads"]
            weights = weights.reshape(heads, 2, -1, weights.shape[-1]).transpose(0, 2, 1, 3).reshape(weights.shape)
        writer.add_tensor(names.get_name(name, try_suffixes=(".weight",)), weights)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def blocks(output):
    rng = random.Random(84193)
    result = {}
    for name in ("Q4_0", "Q8_0"):
        dtype = GGMLQuantizationType[name]
        _, width = GGML_QUANT_SIZES[dtype]
        data = []
        for i, scale in enumerate((0x3800, 0xB400, 0x0001, 0x0000, 0x7BFF, 0x801F)):
            payload = bytes(rng.randrange(256) for _ in range(width - 2))
            if i == 0:
                payload = bytes(range(256))[::17] if name == "Q4_0" else bytes(range(112, 144))
            data.append(scale.to_bytes(2, "little") + payload)
        raw = np.frombuffer(b"".join(data), np.uint8).reshape(-1, width)
        expected = dequantize(raw, dtype).astype("<f4")
        result[name] = {"blocks": [value.hex() for value in data], "fp32_bits": expected.view("<u4").tolist()}
    (output / "blocks.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model(args.output)
    blocks(args.output)
