"""Compare representative real blocks with gguf-py and pinned ggml C decoding."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

from gguf import GGML_QUANT_SIZES, GGUFReader, dequantize
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    bridge = ctypes.CDLL(str(args.bridge.resolve())).nathr_reference_decode
    bridge.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    bridge.restype = ctypes.c_int

    sources = [(args.controlled / entry["filename"], entry)
               for entry in manifest["controlled"]["files"]]
    sources.extend((args.directory / source["repo"] / source["revision"] / entry["filename"], entry)
                   for source in manifest["sources"] for entry in source["files"]
                   if entry["filename"].endswith(".gguf"))
    reports = {}
    for file, entry in sources:
        for tensor in GGUFReader(file).tensors:
            encoding = tensor.tensor_type
            if encoding.name in reports:
                continue
            count, width = GGML_QUANT_SIZES[encoding]
            raw = tensor.data.view(np.uint8).reshape(-1, width)
            blocks = np.ascontiguousarray(raw[[0, len(raw) // 2, len(raw) - 1]])
            if encoding.name in {"F32", "F16"}:
                dtype = "<f4" if encoding.name == "F32" else "<f2"
                expected = blocks.view(dtype).astype(np.float32).reshape(-1)
            else:
                expected = dequantize(blocks, encoding).reshape(-1)
            actual = np.empty(expected.shape, dtype=np.float32)
            called = bridge(int(encoding), blocks.ctypes.data, actual.ctypes.data, actual.size)
            if not called:
                if encoding.name != "F32":
                    raise ValueError(f"no ggml decoder for {encoding.name}")
                # ggml stores F32 natively and has no conversion callback.
                actual[:] = blocks.view("<f4").reshape(-1)
            if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                raise ValueError(f"{encoding.name}: non-finite decoded blocks")
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
            reports[encoding.name] = {
                "type_id": int(encoding), "elements_per_block": count, "bytes_per_block": width,
                "source": file.name, "artifact_sha256": entry["sha256"], "tensor": tensor.name,
                "block_indices": [0, len(raw) // 2, len(raw) - 1],
                "packed_sha256": hashlib.sha256(blocks.tobytes()).hexdigest(),
                "ggml_callback": bool(called),
                "exact_fp32_bits": bool(np.array_equal(actual.view(np.uint32), expected.view(np.uint32))),
                "max_abs": float(np.max(np.abs(actual - expected))),
            }
    missing = manifest["required_gguf_encodings"].keys() - reports.keys()
    if missing:
        raise ValueError(f"missing reference coverage for {sorted(missing)}")
    args.output.write_text(json.dumps(reports, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
