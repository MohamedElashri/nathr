"""Run the complete model correctness suite from the pinned local cache."""

import argparse
import filecmp
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

from models import check


def extended_tokenizers(binary, native, representatives):
    from phase1_verify import check_tokenizer

    fragments = ("", " ", "\t→", "é\u0301", "中", "\u2028", "\ufeff", "\x00")
    added = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<empty_output>")
    extra = [left + word + right for left in fragments for word in added for right in fragments]
    report = {}
    for fingerprint, path in representatives.items():
        report[fingerprint] = check_tokenizer(
            binary, path, Path("fixtures/phase0/tokenizer.json"), native,
            random_cases=10000, seed=59031, extra=extra)
        print("Extended tokenizer differential passed", fingerprint, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "tests", "native", "published", "controlled", "extended", "untied", "llama-probe",
                 "llama-source", "ggml-bridge", "work"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--threads", type=int, choices=range(1, 65), default=4)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}

    def run(script, *options):
        subprocess.run([sys.executable, Path("reference") / script, *options], env=env, check=True)

    def digest(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    manifest_path = Path("models/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    representatives = {"native": args.native}
    files = 0
    for source in manifest["sources"]:
        root = (args.native if source["repo"] == manifest["canonical_model"]
                else args.published / source["repo"] / source["revision"])
        for item in source["files"]:
            if item["required"]:
                path = root / item["filename"]
                check(path, item)
                files += 1
                if path.suffix == ".gguf":
                    representatives[item["inventory"]["tokenizer_fingerprint"]] = path
    for item in manifest["controlled"]["files"]:
        path = args.controlled / item["filename"]
        representatives.setdefault(item["inventory"]["tokenizer_fingerprint"], path)

    fixture = args.work / "fixture-replay"
    run("generate_phase3_fixtures.py", "--bridge", args.ggml_bridge, "--tokenizer", args.native / "tokenizer.json",
        "--output", fixture)
    for name in ("blocks.json", "corpus.json"):
        if not filecmp.cmp(fixture / name, Path("fixtures/phase3") / name, shallow=False):
            raise ValueError(f"replayed fixture differs: {name}")
    header = args.work / "iq_codebooks.hpp"
    run("generate_iq_codebooks.py", args.llama_source / "ggml/src/ggml-common.h", header, "--verify-oracle")
    if not filecmp.cmp(header, "src/quantization/iq_codebooks.hpp", shallow=False):
        raise ValueError("replayed quantization tables differ")

    run("phase1_verify.py", "--binary", args.binary, "--model", args.native,
        "--output", args.work / "native", "--floating-variants")
    run("phase2_verify.py", "--binary", args.binary, "--model", args.native, "--published", args.published,
        "--controlled", args.controlled, "--untied", args.untied, "--output", args.work / "phase2")
    run("phase3_verify.py", "--binary", args.binary, "--native", args.native, "--published", args.published,
        "--controlled", args.controlled, "--extended", args.extended, "--llama-probe", args.llama_probe,
        "--work", args.work / "coverage-work", "--output", args.work / "coverage.json",
        "--threads", str(args.threads), "--workers", str(args.workers))
    run("phase4_verify.py", "--binary", args.binary, "--native", args.native, "--controlled", args.controlled,
        "--output", args.work / "cache.json", "--threads", str(args.threads))

    os.environ.update(env)
    tokenizer_report = extended_tokenizers(args.binary, args.native, representatives)

    subprocess.run([args.tests, "--long"], check=True)
    allocations = subprocess.check_output([args.tests, "--real-allocations", args.controlled / "SmolLM2.Q8_0.gguf"],
                                          text=True)
    coverage = json.loads((args.work / "coverage.json").read_text())
    if len(coverage["artifacts"]) != 58 or coverage["failures"]:
        raise ValueError("incomplete frozen coverage")
    if set(tokenizer_report) - {"native"} != set(coverage["tokenizers"]):
        raise ValueError("incomplete differential tokenizer coverage")
    report = {
        "binary_sha256": digest(args.binary), "manifest_sha256": digest(manifest_path),
        "lock_sha256": digest("reference/requirements.lock"), "python_version": sys.version.split()[0],
        "versions": {name: version(name) for name in ("torch", "transformers", "numpy", "tokenizers", "gguf", "safetensors")},
        "manifest_files_verified": files, "frozen_coverage": len(coverage["artifacts"]),
        "coverage_sha256": digest(args.work / "coverage.json"), "runtime_threads": args.threads,
        "llama_probe_sha256": digest(args.llama_probe), "ggml_bridge_sha256": digest(args.ggml_bridge),
        "untied_gguf_sha256": digest(args.untied),
        "avx2_tested": all(a["structure"]["cpu_features"]["avx2"] for a in coverage["artifacts"]),
        "tokenizer_differential": tokenizer_report, "cache": json.loads((args.work / "cache.json").read_text()),
        "allocation_test": json.loads(allocations), "fixture_replay": "byte-identical",
    }
    (args.work / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
