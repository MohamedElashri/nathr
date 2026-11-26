"""Run complete frozen GGUF coverage and stream the descriptive quality evaluation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from tokenizers import Tokenizer
import torch

from audit_models import inventory
from generate_fixtures import IDS, load_hf
from gguf_oracle import reconstruct
from models import check
from phase1_verify import capture, check_tokenizer, compare
from phase2_verify import audit, execute, file_hash, summary
from rounded_oracle import rounded_operations


EXTENDED = {
    "Q5_1": "ed8c8118f842b4d8e5f5c9fde6dbc8c6d44fb005f73e6481dbfb0cb5729e4897",
    "Q4_K_M": "233255d9088457093ae1123c5c3ce32e203ee58b00168d1e806cb6b188e42790",
    "IQ4_NL": "210397e496e2cdb955082f6c89d9060ad9423ea185b75804b4950c55f3a11d07",
}


def corpus_windows(path, native):
    raw = path.read_bytes()
    source = json.loads(raw)
    tokenizer_path = native / "tokenizer.json"
    if file_hash(tokenizer_path) != source["tokenizer_sha256"]:
        raise ValueError("corpus tokenizer checksum differs")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    context, stride = source["context"], source["stride"]
    if not 0 < stride <= context <= 128 or source["bos"] != 0:
        raise ValueError("invalid corpus context, stride or BOS")
    windows = []
    for index, doc in enumerate(source["documents"]):
        current = tokenizer.encode(doc["text"], add_special_tokens=False).ids
        if current != doc["ids"]:
            raise ValueError(f"document {index}: token IDs differ from the pinned fixture")
        sequence = [source["bos"], *current]
        covered = 0
        for start in range(0, len(current), stride):
            stop = min(start + context, len(current))
            skip = covered - start
            if not 0 <= skip < stop - start:
                raise ValueError("corpus window did not advance the target sequence")
            windows.append({
                "doc": index, "start": start, "skip": skip,
                "ids": sequence[start:stop], "targets": sequence[start+1:stop+1],
            })
            covered = stop
            if stop == len(current):
                break
        if covered != len(current):
            raise ValueError("corpus target coverage is incomplete")
    return windows, {
        "sha256": hashlib.sha256(raw).hexdigest(), "tokenizer_sha256": source["tokenizer_sha256"],
        "documents": len(source["documents"]), "context": context, "stride": stride, "bos": source["bos"],
        "windows": [{"document": w["doc"], "start": w["start"], "size": len(w["ids"]), "unscored_overlap": w["skip"]}
                    for w in windows],
        "scored_tokens": sum(len(w["ids"]) - w["skip"] for w in windows),
    }


def log_probabilities(logits):
    row = logits.astype(np.float64)
    if not np.isfinite(row).all():
        raise ValueError("non-finite corpus logits")
    row -= row.max()
    return row - np.log(np.exp(row).sum())


def top5(row):
    boundary = np.partition(row, -5)[-5]
    above = np.flatnonzero(row > boundary)
    ties = np.flatnonzero(row == boundary)
    return np.concatenate((above, ties[:5 - above.size]))


@torch.inference_mode()
def original_windows(native, windows):
    model = load_hf(native)
    result, nll, tokens = [], 0.0, 0
    for w in windows:
        tensor = torch.tensor([w["ids"]], dtype=torch.long)
        logits = model(tensor, use_cache=False).logits[0].numpy()
        rows = []
        for target, row in zip(w["targets"][w["skip"]:], logits[w["skip"]:], strict=True):
            logp = log_probabilities(row)
            nll -= logp[target]
            tokens += 1
            rows.append((logp, np.exp(logp), int(np.argmax(row)), top5(row)))
        result.append(rows)
    return result, {"scored_tokens": tokens, "mean_nll": nll / tokens, "perplexity": math.exp(nll / tokens)}


def quality(rows):
    nll = kl = 0.0
    top1 = topk = tokens = 0
    for row, target, base in rows:
        logp = log_probabilities(row)
        expected_logp, expected_prob, expected_top1, expected_top5 = base
        nll -= logp[target]
        kl += np.sum(expected_prob * (expected_logp - logp))
        top1 += int(np.argmax(row) == expected_top1)
        topk += len(np.intersect1d(top5(row), expected_top5, assume_unique=True))
        tokens += 1
    return {
        "scored_tokens": tokens, "mean_nll": nll / tokens, "perplexity": math.exp(nll / tokens),
        "kl_native_to_artifact": max(0.0, kl / tokens), "top1_agreement": top1 / tokens,
        "mean_top5_overlap": topk / (tokens * 5),
    }


def model_rows(model, windows, original):
    with torch.inference_mode():
        for window, base in zip(windows, original, strict=True):
            token_tensor = torch.tensor([window["ids"]], dtype=torch.long)
            logits = model(token_tensor, use_cache=False).logits[0].numpy()
            skip = window["skip"]
            yield from zip(logits[skip:], window["targets"][skip:], base, strict=True)


def stream_rows(binary, path, window, base, options=()):
    vocab = base[0][0].size
    cmd = [binary, "forward", "--model", path, "--backend", "auto", "--cached",
           "--context", str(len(window["ids"])), "--logits-f32", "--tokens", ",".join(map(str, window["ids"])),
           *options]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as process:
        for i, target in enumerate(window["targets"]):
            raw = process.stdout.read(4 * vocab)
            if len(raw) != 4 * vocab:
                raise RuntimeError(f"{path}: incomplete streamed logit row {i}")
            if i >= window["skip"]:
                yield np.frombuffer(raw, dtype="<f4"), target, base[i - window["skip"]]
        if process.stdout.read(1) or process.wait() != 0:
            raise RuntimeError(f"{path}: scoring stream returned extra bytes or failed")


def verify_stream(binary, artifact, model, windows, original, threads=1):
    # Use the second window so the stream test also exercises overlap masking.
    window, baseline = windows[1], original[1]
    maximum = 0.0
    for actual, expected in zip(stream_rows(binary, artifact, window, baseline, ("--threads", str(threads))),
                                model_rows(model, [window], [baseline]), strict=True):
        got, got_target, _ = actual
        ref, ref_target, _ = expected
        delta = np.linalg.norm(got.astype(np.float64) - ref) / max(np.linalg.norm(ref.astype(np.float64)), 1e-12)
        if got_target != ref_target or not np.isfinite(delta) or delta > 1e-4:
            raise ValueError(f"{artifact}: streamed logits differ at target {got_target}, relative L2={delta}")
        maximum = max(maximum, delta)
    return {"rows": len(baseline), "masked_overlap": window["skip"], "max_relative_l2": maximum}


def initialize_worker(args, manifest, windows, original):
    global run_args, run_manifest, run_windows, run_original
    run_args, run_manifest, run_windows, run_original = args, manifest, windows, original
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def extended_item(path, manifest):
    tables = {name: {} for name in ("shape_inventories", "encoding_inventories", "architectures", "tokenizers")}
    recipe = path.stem.removeprefix("SmolLM2.")
    digest = file_hash(path)
    if digest != EXTENDED[recipe]:
        raise ValueError(f"{path}: controlled extension checksum differs")
    data = inventory(path, tables)
    item = {"filename": path.name, "size": path.stat().st_size, "sha256": digest, "inventory": data}
    reference = {**manifest, **tables}
    return item, reference


def qualify(job):
    suite, artifact, item = job
    args = run_args
    manifest = run_manifest
    if item is None:
        item, manifest = extended_item(artifact, manifest)
    structure = audit(args.binary, artifact, item, manifest, "auto")
    if not structure["all_encodings_executable"]:
        raise ValueError(f"{artifact}: incomplete execution registry")
    avx2 = structure["cpu_features"]["avx2"]
    with TemporaryDirectory(prefix="artifact-", dir=args.work) as staging:
        root = Path(staging)
        model, _ = reconstruct(artifact, args.native)
        eager = capture(model, IDS)
        descriptive = quality(model_rows(model, run_windows, run_original))
        with rounded_operations(model):
            expected = capture(model, IDS)
            streamed = None
            if suite == "controlled_extension":
                streamed = verify_stream(args.binary, artifact, model, run_windows, run_original, args.threads)
        violations = {}
        parity = execute(args.binary, artifact, args.native, root, avx2, model=model, measure_rss=True,
                         violations=violations, expected=expected, threads=args.threads)
        baseline = parity["oracle_greedy_ids"]
        eager_ids = eager["logits"].argmax(axis=-1).tolist()
        if "independent_fp32_greedy_ids" in item and item["independent_fp32_greedy_ids"] != eager_ids:
            raise ValueError(f"{artifact}: pinned reference IDs differ")
        if baseline != eager_ids:
            raise ValueError(f"{artifact}: rounded and eager oracle greedy IDs differ: {baseline} vs {eager_ids}")
        drift = []
        eager_vs_rounded = summary(compare(expected, eager, drift))
        external = subprocess.run([args.llama_probe, "forward", artifact, *map(str, IDS)],
                                  text=True, capture_output=True)
        if external.returncode:
            raise RuntimeError(f"{artifact}: llama.cpp cross-check failed: {external.stderr[-800:]}")
        external_ids = json.loads(external.stdout)
        agreement = sum(a == b for a, b in zip(baseline, external_ids, strict=True))
    return {
        "suite": suite, "filename": artifact.name, "sha256": item["sha256"],
        "recipe": item.get("recipe", artifact.stem.removeprefix("SmolLM2.")),
        "dtype_histogram": item["inventory"]["dtype_histogram"],
        "effective_bits_per_stored_parameter": item["inventory"]["effective_bits_per_stored_parameter"],
        "tokenizer_fingerprint": item["inventory"]["tokenizer_fingerprint"],
        "structure": structure, "parity": parity, "quality_vs_native": descriptive,
        "eager_fp32_vs_rounded_oracle": {**eager_vs_rounded, "threshold_crossings": drift},
        "streamed_logits_check": streamed,
        "gate_violations": violations,
        "llama_cpp": {"greedy_ids": external_ids, "exact_greedy_matches": agreement},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "native", "published", "controlled", "extended", "llama-probe", "work", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--corpus", type=Path, default=Path("fixtures/phase3/corpus.json"))
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--threads", type=int, choices=range(1, 65), default=1)
    parser.add_argument("--artifact", action="append", metavar="FILENAME",
                        help="restrict the matrix to one or more exact GGUF filenames")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    jobs, tokenizers = [], {}
    for source in manifest["sources"]:
        for item in source["files"]:
            if item["filename"].endswith(".gguf"):
                path = args.published / source["repo"] / source["revision"] / item["filename"]
                jobs.append((source["repo"], path, item))
                tokenizers[item["inventory"]["tokenizer_fingerprint"]] = path
            elif source["repo"] == manifest["canonical_model"] and item["required"]:
                check(args.native / item["filename"], item)
    for item in manifest["controlled"]["files"]:
        path = args.controlled / item["filename"]
        jobs.append(("controlled", path, item))
        tokenizers[item["inventory"]["tokenizer_fingerprint"]] = path
    for recipe in EXTENDED:
        jobs.append(("controlled_extension", args.extended / f"SmolLM2.{recipe}.gguf", None))
    if args.artifact:
        requested = set(args.artifact)
        known = {path.name for _, path, _ in jobs}
        if requested - known:
            parser.error(f"unknown artifact filenames: {sorted(requested - known)}")
        jobs = [job for job in jobs if job[1].name in requested]
    torch.set_num_threads(1)
    windows, corpus = corpus_windows(args.corpus, args.native)
    original, native_quality = original_windows(args.native, windows)
    report = {
        "schema_version": 2, "corpus": corpus, "native_quality": native_quality, "artifacts": [],
        "parity_reference": "independent PyTorch with FP64 intermediates and FP32 activation boundaries",
        "runtime_threads": args.threads,
        "quality_engine": "independent eager FP32 PyTorch using exactly decoded artifact weights",
        "binary_sha256": file_hash(args.binary), "manifest_sha256": file_hash(args.manifest),
        "llama_cpp": {"revision": manifest["controlled"]["converter_revision"],
                      "role": "additional interoperability, greedy agreement only",
                      "note": "llama.cpp may use quantized activations"},
        "controlled_extension": {
            "source_sha256": next(f["sha256"] for f in manifest["controlled"]["files"] if f["recipe"] == "f32"),
            "quantizer_revision": manifest["controlled"]["quantizer_revision"],
            "command": "llama-quantize INPUT_F32 OUTPUT RECIPE 3",
            "recipes": EXTENDED, "importance_matrix": None,
        },
        "tokenizers": {digest: check_tokenizer(args.binary, path, Path("fixtures/phase0/tokenizer.json"),
                                               args.native) for digest, path in tokenizers.items()},
    }
    errors = []
    if args.artifact:
        report["selected_filenames"] = sorted(requested)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker, initargs=(args, manifest, windows, original)) as pool:
        running = {pool.submit(qualify, job): (idx, job) for idx, job in enumerate(jobs)}
        results = {}
        for future in as_completed(running):
            index, (suite, path, _) = running[future]
            try:
                result = future.result()
                result["tokenizer_status"] = "matched_reference" if result["tokenizer_fingerprint"] in report["tokenizers"] else "unverified"
                if result["tokenizer_status"] != "matched_reference":
                    raise ValueError(f"{path}: tokenizer fingerprint has no independent proof")
                results[index] = result
                violations = result["gate_violations"]
                if violations:
                    errors.append({"suite": suite, "filename": path.name,
                                   "error": "same-artifact numerical gate exceeded", "paths": sorted(violations)})
                status = "failed" if violations else "qualified"
                print(f"{status} {len(results)}/{len(jobs)}: {suite}/{path.name}", flush=True)
            except Exception as error:
                errors.append({"suite": suite, "filename": path.name, "error": str(error)})
                print(f"failed {suite}/{path.name}: {error}", flush=True)
            report["artifacts"] = [results[i] for i in sorted(results)]
            report["failures"] = errors
            temporary = args.output.with_suffix(".partial")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.output)
    if errors:
        raise SystemExit(f"{len(errors)} artifacts failed validation")


if __name__ == "__main__":
    main()
