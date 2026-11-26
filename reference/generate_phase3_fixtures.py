"""Generate adversarial packed blocks and a small fixed Phase 3 quality corpus."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import random

from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, dequantize
import numpy as np
from tokenizers import Tokenizer


PACKED = (
    "Q4_1", "Q5_0", "Q5_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS",
)
SCALE_LOCATIONS = {
    "Q4_1": (0, 2), "Q5_1": (0, 2), "Q2_K": (80, 82), "Q3_K": (108,),
    "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,),
}
SCALES = (0x3C00, 0xBC00, 0x0001, 0x8001, 0x0000, 0x8000, 0x7BFF, 0x0400)
DOCUMENTS = (
    "Mira carried a small notebook through the rain. At the station she drew a map of the river, "
    "then watched the last train leave as the lamps came on across the bridge.",
    "The terminal printed 42 before it stopped.\n"
    "for value in [1, 2, 3]:\n    print(value * value)\n"
    "The loop writes one square per line.",
    "At the café, Sam heard مرحبا and answered with a smile. "
    "On the menu someone had written 你好 beside a drawing of a teapot.",
)


def c_decoder(path):
    decode = ctypes.CDLL(str(path.resolve())).nathr_reference_decode
    decode.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    decode.restype = ctypes.c_int
    return decode


def blocks(path, bridge):
    rng = random.Random(1497835)
    decoder = c_decoder(bridge)
    result = {}
    for kind in PACKED:
        dtype = GGMLQuantizationType[kind]
        n, size = GGML_QUANT_SIZES[dtype]
        raw_blocks = []
        for index, scale in enumerate(SCALES):
            if index < 2:
                raw = bytearray([255 * index] * size)
            elif index < 4:
                raw = bytearray((i * (23 + index) + 17) % 256 for i in range(size))
            else:
                raw = bytearray(rng.randbytes(size))
            if kind == "IQ1_M":
                for i in range(4):
                    raw[49 + i * 2] &= 15
                    raw[49 + i * 2] |= ((scale >> (i * 4)) & 15) << 4
            else:
                for index_in_block, offset in enumerate(SCALE_LOCATIONS.get(kind, (0,))):
                    value = scale if not index_in_block else SCALES[(index + 3) % len(SCALES)]
                    raw[offset:offset + 2] = value.to_bytes(2, "little")
            raw_blocks.append(bytes(raw))
        raw = np.frombuffer(b"".join(raw_blocks), np.uint8).reshape(-1, size)
        oracle = np.ascontiguousarray(dequantize(raw, dtype).astype("<f4"))
        reference = np.empty_like(oracle)
        if not decoder(int(dtype), raw.ctypes.data, reference.ctypes.data, raw.shape[0] * n):
            raise ValueError(f"{kind}: pinned GGML decoder is missing")
        if not np.isfinite(oracle).all():
            raise ValueError(f"{kind}: synthesized invalid or non-finite weights")
        np.testing.assert_array_equal(oracle.view("<u4"), reference.view("<u4"), err_msg=kind)
        result[kind] = {
            "blocks": [block.hex() for block in raw_blocks],
            "fp32_le": [row.tobytes().hex() for row in oracle],
        }
    path.write_text(json.dumps(result, indent=2) + "\n")


def corpus(path, tokenizer_path):
    data = tokenizer_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c":
        raise ValueError("expected the pinned SmolLM2 tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    result = {
        "schema_version": 1, "tokenizer_sha256": digest,
        "bos": 0, "context": 32, "stride": 24,
        "documents": [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=False).ids}
                      for text in DOCUMENTS],
    }
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    blocks(args.output / "blocks.json", args.bridge)
    corpus(args.output / "corpus.json", args.tokenizer)
