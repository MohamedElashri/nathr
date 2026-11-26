"""Audit the frozen GGUF inventory and validate the v0.2 execution slice."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from gguf import GGUFReader
import numpy as np
from safetensors.numpy import load_file
import torch

from generate_fixtures import IDS
from gguf_oracle import reconstruct
from models import check
from phase1_verify import capture, check_tokenizer, command, compare


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(binary, path, item, manifest, backend="scalar"):
    check(path, item)
    actual = json.loads(command(binary, "inspect", path, "--backend", backend))
    expected = item["inventory"]
    for key in ("format", "version", "byte_order", "alignment", "data_offset", "output_tensor",
                "split", "stored_parameters", "encoded_weight_bytes", "tokenizer_fingerprint"):
        if actual[key] != expected[key]:
            raise ValueError(f"{path}: mismatch in {key}")
    if actual["file_bytes"] != item["size"] or actual["owned_file_bytes"] != 0:
        raise ValueError(f"{path}: wrong file size or mapped storage policy")
    if actual["mapped_weight_bytes"] != expected["encoded_weight_bytes"]:
        raise ValueError(f"{path}: wrong mapped weight bytes")
    if actual["architecture"] != manifest["architectures"][expected["architecture_fingerprint"]]:
        raise ValueError(f"{path}: architecture metadata mismatch")
    expected_shapes = manifest["shape_inventories"][expected["shape_inventory"]]
    expected_kinds = manifest["encoding_inventories"][expected["encoding_inventory"]]
    types = {name: kind for kind, names in expected_kinds.items() for name in names}
    report = actual["tensor_table"]
    if report.keys() != expected_shapes.keys() or report.keys() != types.keys():
        raise ValueError(f"{path}: tensor name set differs")
    for name, descriptor in report.items():
        kind = types[name]
        if (descriptor["shape"] != expected_shapes[name] or descriptor["encoding"] != kind or
                descriptor["id"] != expected["dtype_histogram"][kind]["id"]):
            raise ValueError(f"{path}: tensor layout differs for {name}")
        block = manifest["reference_checks"]["block_decoder_cross_checks"][kind]
        size = int(np.prod(descriptor["shape"])) // block["elements_per_block"] * block["bytes_per_block"]
        if descriptor["encoded_bytes"] != size:
            raise ValueError(f"{path}: tensor byte size differs for {name}")
    for kind, summary in expected["dtype_histogram"].items():
        if actual["dtype_histogram"][kind] != {key: summary[key] for key in ("tensors", "parameters", "bytes")}:
            raise ValueError(f"{path}: encoding histogram differs for {kind}")
    return {
        "filename": path.name, "sha256": item["sha256"], "types": sorted(expected_kinds),
        "all_encodings_executable": actual["all_encodings_executable"],
        "cpu_features": actual["cpu_features"],
        "linear_kernels": actual["linear_kernels"],
        "mapped_file_bytes": actual["mapped_file_bytes"],
        "encoded_weight_bytes": actual["encoded_weight_bytes"],
        "owned_file_bytes": actual["owned_file_bytes"],
    }


def reject(command_line, error):
    result = subprocess.run(command_line, text=True, capture_output=True)
    if result.returncode == 0 or error not in result.stderr:
        raise ValueError(f"expected {error!r}, got {result.returncode} / {result.stderr}")


def tokenizer_sidecar(binary, model, canonical, output):
    copy = output / "alternate-tokenizer.gguf"
    shutil.copyfile(model, copy)
    reader = GGUFReader(model)
    pre = reader.fields["tokenizer.ggml.pre"]
    first_token = reader.fields["tokenizer.ggml.tokens"]
    absolute = lambda field, index: field.parts[field.data[index]].ctypes.data - reader.data.ctypes.data
    with copy.open("r+b") as stream:
        stream.seek(absolute(pre, 0))
        stream.write(b"otherm")
    prefix = [binary, "tokenize", "--model", copy, "--prompt", "مرحبا <|im_end|> 42"]
    reject(prefix, "unknown GGUF tokenizer fingerprint")
    original = command(binary, "tokenize", model, "--prompt", prefix[-1])
    approved = canonical / "tokenizer.json"
    override = command(binary, "tokenize", copy, "--prompt", prefix[-1], "--tokenizer", approved)
    if original != override:
        raise ValueError("pinned sidecar changed tokenization")
    reject([*prefix, "--tokenizer", "fixtures/phase1/tokenizer.json"], "expected the pinned SmolLM2 tokenizer.json")
    with copy.open("r+b") as stream:
        stream.seek(absolute(first_token, 0))
        stream.write(b"!")
    reject([*prefix, "--tokenizer", approved], "GGUF vocabulary differs")
    copy.unlink()


def summary(metrics):
    worst = max(metrics, key=lambda name: metrics[name]["relative_l2"])
    return {"tensors": len(metrics), "worst_tensor": worst, "worst_metrics": metrics[worst], "logits": metrics["logits"]}


def execute(binary, path, native, root, avx2, model=None, measure_rss=False, violations=None, expected=None,
            threads=1):
    if model is None:
        model, _ = reconstruct(path, native)
    if expected is None:
        expected = capture(model, IDS)
    trace_dir = root / path.stem
    trace_dir.mkdir(exist_ok=True)
    result = {}

    def verify(name, reference, actual):
        current = [] if violations is not None else None
        try:
            result[name] = summary(compare(reference, actual, current))
        except (AssertionError, ValueError) as error:
            raise ValueError(f"{name}: {error}") from error
        if current:
            violations[name] = current

    for backend in (("scalar", "avx2") if avx2 else ("scalar",)):
        for cached in (False, True):
            name = backend + ("-cached" if cached else "")
            trace = trace_dir / f"{name}.safetensors"
            options = ["--tokens", ",".join(map(str, IDS)), "--trace", trace, "--backend", backend,
                       "--threads", str(threads)]
            if cached:
                options.extend(["--cached", "--context", "512"])
            if measure_rss:
                usage = trace_dir / "rss"
                subprocess.run(["/usr/bin/time", "-f", "%M", "-o", usage, binary, "forward", "--model", path,
                                *options], capture_output=True, check=True, text=True)
            else:
                command(binary, "forward", path, *options)
            actual = load_file(trace)
            verify(name, expected, actual)
            if measure_rss:
                result[name]["peak_rss_kib"] = int(usage.read_text())
            if cached:
                for key in actual:
                    np.testing.assert_array_equal(actual[key], uncached[key])
            else:
                uncached = actual
                if backend == "scalar":
                    scalar = actual
                else:
                    verify("avx2_vs_scalar", scalar, actual)
    result["separate_output"] = not model.config.tie_word_embeddings
    result["oracle_greedy_ids"] = expected["logits"].argmax(axis=-1).tolist()
    result["sha256"] = file_hash(path)
    print(f"compared {path.name}: {len(expected)} tensors per path", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--untied", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    inspected, token_models = [], {}
    for source in manifest["sources"]:
        for item in source["files"]:
            if item["filename"].endswith(".gguf"):
                file = args.published / source["repo"] / source["revision"] / item["filename"]
                inspected.append(audit(args.binary, file, item, manifest))
                token_models[item["inventory"]["tokenizer_fingerprint"]] = file
            elif source["repo"] == manifest["canonical_model"] and item["required"]:
                check(args.model / item["filename"], item)
    files = []
    for item in manifest["controlled"]["files"]:
        file = args.controlled / item["filename"]
        inspected.append(audit(args.binary, file, item, manifest))
        token_models[item["inventory"]["tokenizer_fingerprint"]] = file
        files.append(file)
    avx2 = inspected[-1]["cpu_features"]["avx2"]
    print(f"inspected {len(inspected)} frozen files and controlled conversions", flush=True)
    results = {file.name: execute(args.binary, file, args.model, args.output, avx2)
               for file in [*files, args.untied]}
    tokenizers = {fingerprint: check_tokenizer(args.binary, file, Path("fixtures/phase0/tokenizer.json"), args.model)
                  for fingerprint, file in token_models.items()}
    tokenizer_sidecar(args.binary, files[-1], args.model, args.output)
    report = {"inspected": inspected, "runs_avx2": avx2, "execution": results,
              "tokenizers": tokenizers, "sidecar": "validated canonical hash and rejected mismatched tables"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
