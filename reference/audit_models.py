"""Audit local artifact bytes and tensor tables with Safetensors and GGUF readers."""

import argparse
from collections import defaultdict
import hashlib
import json
from math import prod
from pathlib import Path
import struct

from gguf import GGML_QUANT_SIZES, GGUFReader
from safetensors import safe_open

from models import check


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def safetensors_inventory(path):
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
    header.pop("__metadata__", None)
    tensors = []
    with safe_open(path, framework="np") as reader:
        for name in reader.keys():
            tensor = reader.get_slice(name)
            begin, end = header[name]["data_offsets"]
            tensors.append((name, tensor.get_dtype(), None, tensor.get_shape(), end - begin))
    architecture = json.loads(path.with_name("config.json").read_text())
    files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    tokenizer = {name: fingerprint(json.loads(path.with_name(name).read_text())) for name in files}
    return tensors, architecture, tokenizer, {
        "format": "safetensors", "header_bytes": length, "byte_order": "little",
        "output_tensor": "lm_head.weight" in header,
    }


def gguf_inventory(path):
    reader = GGUFReader(path)
    fields = {key: field.contents() for key, field in reader.fields.items()}
    architecture = {key: val for key, val in fields.items() if key.startswith("llama.")}
    architecture["general.architecture"] = fields.get("general.architecture")
    tokenizer = {key: val for key, val in fields.items() if key.startswith("tokenizer.")}
    if architecture["general.architecture"] != "llama":
        raise ValueError(f"{path}: expected GGUF Llama")
    file_size = path.stat().st_size
    tensors = []
    for tensor in reader.tensors:
        name, encoding = tensor.name, tensor.tensor_type
        block, size = GGML_QUANT_SIZES[encoding]
        shape = list(map(int, tensor.shape))  # GGUF order, contiguous dimension first
        if shape[0] % block or prod(shape) // block * size != tensor.n_bytes:
            raise ValueError(f"{path}: incompatible block geometry for {name}")
        if tensor.data_offset % reader.alignment or tensor.data_offset + tensor.n_bytes > file_size:
            raise ValueError(f"{path}: incompatible location for {name}")
        tensors.append((name, encoding.name, int(encoding), shape, int(tensor.n_bytes)))
    return tensors, architecture, tokenizer, {
        "format": "gguf",
        "version": fields["GGUF.version"],
        "byte_order": reader.endianess.name.lower(),
        "alignment": reader.alignment,
        "data_offset": reader.data_offset,
        "output_tensor": any(tensor.name == "output.weight" for tensor in reader.tensors),
        "split": {key: val for key, val in fields.items() if key.startswith("split.")},
    }


def put(table, value):
    key = fingerprint(value)
    table[key] = value
    return key


def inventory(path, tables):
    read = gguf_inventory if path.suffix == ".gguf" else safetensors_inventory
    tensors, architecture, tokenizer, result = read(path)
    shapes = dict(sorted((name, shape) for name, _, _, shape, _ in tensors))
    if len(shapes) != len(tensors):
        raise ValueError(f"{path}: duplicate tensor names")
    types = {}
    placement = defaultdict(list)
    for name, encoding, type_id, shape, size in sorted(tensors):
        entry = types.setdefault(encoding, {"id": type_id, "tensors": 0, "parameters": 0, "bytes": 0})
        entry["tensors"] += 1
        entry["parameters"] += prod(shape)
        entry["bytes"] += size
        placement[encoding].append(name)

    parameters = sum(entry["parameters"] for entry in types.values())
    encoded = sum(entry["bytes"] for entry in types.values())
    result.update(
        stored_parameters=parameters,
        encoded_weight_bytes=encoded,
        effective_bits_per_stored_parameter=8 * encoded / parameters,
        dtype_histogram=types,
        shape_inventory=put(tables["shape_inventories"], shapes),
        encoding_inventory=put(tables["encoding_inventories"], dict(placement)),
        architecture_fingerprint=put(tables["architectures"], architecture),
        tokenizer_fingerprint=fingerprint(tokenizer),
    )
    key = result["tokenizer_fingerprint"]
    if result["format"] == "gguf":
        tokenizer = {
            field: {"length": len(value), "sha256": fingerprint(value)} if isinstance(value, list) else value
            for field, value in tokenizer.items()
        }
    tables["tokenizers"][key] = tokenizer
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    tables = {name: {} for name in ("shape_inventories", "encoding_inventories", "architectures", "tokenizers")}
    gguf_encodings = {}
    for source in manifest["sources"]:
        for item in source["files"]:
            if not item["required"]:
                continue
            path = args.directory / source["repo"] / source["revision"] / item["filename"]
            item["sha256"] = check(path, item)
            item["verification"] = "local_bytes_verified"
            if item["kind"] == "model":
                item["inventory"] = inventory(path, tables)
                if item["inventory"]["format"] == "gguf":
                    gguf_encodings.update(
                        (name, entry["id"]) for name, entry in item["inventory"]["dtype_histogram"].items()
                    )
    for entry in manifest["controlled"]["files"]:
        path = args.controlled / entry["filename"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if entry["sha256"] != digest:
            raise ValueError(f"{path}: controlled SHA-256 mismatch")
        entry.update(size=path.stat().st_size, inventory=inventory(path, tables))
    manifest["required_gguf_encodings"] = dict(sorted(gguf_encodings.items(), key=lambda pair: pair[1]))
    manifest.update(tables)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
