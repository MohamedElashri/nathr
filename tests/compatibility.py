"""Check pinned, locally downloaded model files and optionally measure mixed CPU generation."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_files(path, profile):
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    checkpoint = next(item for item in manifest["checkpoints"]
                      if profile in [item["name"], *item.get("aliases", [])])
    source = next(item for item in manifest["sources"] if item["repository"] == checkpoint["repository"])
    files = source["files"]
    checkpoint_file = checkpoint.get("file")
    if checkpoint_file:
        require(path.name == checkpoint_file, f"{path}: expected checkpoint file {checkpoint_file}")
        files = [item for item in files if item["filename"] == checkpoint_file]
    require(files, f"{profile}: no files in manifest")
    for item in files:
        filename = path if checkpoint_file else path / item["filename"]
        sha = hashlib.sha256()
        with filename.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        require(sha.hexdigest() == item["sha256"], f"{filename}: SHA-256 mismatch")


def run(binary, *args, requests=None):
    lines = None if requests is None else "".join(json.dumps(item) + "\n" for item in requests)
    result = subprocess.run([binary, *args], input=lines, text=True, capture_output=True, check=True, timeout=180)
    if requests is None:
        return json.loads(result.stdout)
    events = [json.loads(line) for line in result.stdout.splitlines()]
    for event in events:
        require(event["event"] != "error" and event.get("reason") != "error", event)
    return events


def csv(values):
    return ",".join(map(str, values))


def check(binary, path, case, backend):
    pair = ["--text-pair", case["pair"]] if "pair" in case else []
    tokens = run(binary, "tokenize", path, case["prompt"], *pair)
    require(tokens["ids"] == case["ids"], "token IDs diverged")
    if pair:
        require(tokens["token_type_ids"] == case["token_type_ids"], "token type IDs diverged")

    prefix = ["--decoder-tokens", csv(case["decoder_tokens"])] if "decoder_tokens" in case else []
    result = run(binary, "infer", path, case["prompt"], *pair, *prefix, "--output", case["output"],
                 "--backend", backend)
    require(result["shape"] == case["shape"] and len(result["values"]) == math.prod(case["shape"]),
            "tensor shape diverged")
    for offset, expected in case["probes"].items():
        value = result["values"][int(offset)]
        require(math.isclose(value, expected, rel_tol=0, abs_tol=case["tolerance"]),
                f"{case['output']}[{offset}]: {value} differs from {expected}")

    if "last_token_argmax" in case:
        last = result["values"][-result["shape"][-1]:]
        require(max(range(len(last)), key=last.__getitem__) == case["last_token_argmax"],
                "last-token greedy ID diverged")

    if "seeded_ids" in case or "greedy_ids" in case:
        greedy = "greedy_ids" in case
        expected = case["greedy_ids" if greedy else "seeded_ids"]
        sampling = ["--greedy"] if greedy else ["--sample", "--seed", "7", "--top-k", "20",
                                                  "--temperature", "0.8", "--top-p", "0.95"]
        standalone = run(binary, "generate", path, "--tokens", csv(case["ids"]), "--max-new-tokens", "8",
                         *sampling, "--backend", backend, "--json")
        request = dict(tokens=case["ids"], max_new_tokens=8, greedy=greedy)
        if not greedy:
            request.update(seed=7, top_k=20, temperature=0.8, top_p=0.95)
        events = run(binary, "batch", path, "--backend", backend, "--max-active", "2",
                     requests=[dict(request, id=1), dict(request, id=2)])
        require(standalone["generated_ids"] == expected, "standalone generation IDs diverged")
        for req_id in (1, 2):
            require([event["token_id"] for event in events
                     if event["id"] == req_id and event["event"] == "token"] == expected,
                    f"batch generation IDs diverged for request {req_id}")

    if chat := case.get("chat"):
        with tempfile.TemporaryDirectory() as directory:
            messages = Path(directory) / "messages.json"
            messages.write_text(json.dumps(chat["messages"]))
            result = run(binary, "generate", path, "--messages", messages, "--greedy", "--max-new-tokens", "20",
                         "--backend", backend, "--json")
        for field in ("prompt_ids", "generated_ids", "text"):
            require(result[field] == chat[field], f"two-turn chat {field} diverged")
        events = run(binary, "batch", path, "--backend", backend,
                     requests=[dict(id=1, messages=chat["messages"], max_new_tokens=20)])
        require([event["token_id"] for event in events if event["event"] == "token"] == chat["generated_ids"],
                "two-turn batch chat IDs diverged")
        require("".join(event.get("text", "") for event in events) == chat["text"],
                "two-turn batch chat text diverged")


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def measure(binary, path, backend, threads, repetitions, ids):
    requests = [
        dict(id=index, tokens=(ids * math.ceil(length / len(ids)))[:length], max_new_tokens=8)
        for index, length in enumerate((64, 4, 96, 4, 32, 4), start=1)
    ]
    for active in (1, 3):
        samples = []
        for index in range(repetitions + 1):
            events = run(binary, "batch", path, "--backend", backend, "--threads", str(threads),
                         "--max-active", str(active), "--context", "128", "--prefill-chunk", "8",
                         "--step-tokens", "16", "--stats", requests=requests)
            finished = {event["id"] for event in events if event["event"] == "done"}
            require(finished == {request["id"] for request in requests}, "mixed workload did not drain")
            first = [event["elapsed_ms"] for event in events if event["event"] == "token"
                     and event["emitted"] == 1 and event["id"] % 2 == 0]
            intervals = [event["inter_token_ms"] for event in events if "inter_token_ms" in event]
            require(len(first) == 3 and intervals, "mixed workload omitted timing events")
            if index:
                stats = events[-1]
                samples.append(dict(short_ttft_ms=statistics.median(first), p95_itl_ms=p95(intervals),
                                    output_tokens_s=stats["output_tokens_s"], peak_kv_bytes=stats["peak_kv_bytes"],
                                    resident_kv_bytes=stats["resident_kv_bytes"]))
        print(json.dumps(dict(active=active, threads=threads, backend=backend, repetitions=repetitions,
                              short_ttft_ms=statistics.median(x["short_ttft_ms"] for x in samples),
                              p95_itl_ms=statistics.median(x["p95_itl_ms"] for x in samples),
                              output_tokens_s=statistics.median(x["output_tokens_s"] for x in samples),
                              peak_kv_bytes=max(x["peak_kv_bytes"] for x in samples),
                              resident_kv_bytes=max(x["resident_kv_bytes"] for x in samples))))


def main():
    cases = json.loads(Path(__file__).with_name("checkpoints.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*", metavar="CHECKPOINT",
                        help=f"one or more of {', '.join(cases)}, or all (default: gguf)")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/release/nathr")
    parser.add_argument("--models", type=Path, default=ROOT / "models/cache")
    parser.add_argument("--backend", choices=("auto", "scalar", "avx2"), default="auto")
    parser.add_argument("--measure", action="store_true", help="also measure a fixed mixed GGUF workload")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    options = parser.parse_args()
    selected = options.checkpoints or ["gguf"]
    if selected == ["all"]:
        selected = list(cases)
    if set(selected) - cases.keys():
        parser.error("unknown checkpoint selection")
    if not 1 <= options.threads <= 64 or options.repetitions < 1:
        parser.error("expected 1–64 threads and at least one repetition")

    for name in selected:
        case = cases[name]
        path = options.models / case["path"]
        try:
            verify_files(path, name)
            check(options.binary.resolve(), path, case, options.backend)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            parser.exit(1, f"{name}: {getattr(error, 'stderr', None) or error}\n")
        print(f"{name}: passed", flush=True)
    if options.measure:
        gguf = options.models / cases["gguf"]["path"]
        try:
            if "gguf" not in selected:
                verify_files(gguf, "gguf")
            measure(options.binary.resolve(), gguf, options.backend, options.threads, options.repetitions,
                    cases["gguf"]["ids"])
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            parser.exit(1, f"measurement: {getattr(error, 'stderr', None) or error}\n")


if __name__ == "__main__":
    main()
