"""Run bounded, model-free defensive parser and codec fuzz checks with Clang."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
from time import monotonic


def seeds():
    fixtures = Path("fixtures")
    items = {
        "safetensors": {"tiny": (fixtures / "phase1/f32/model.safetensors").read_bytes()},
        "gguf": {"tiny": (fixtures / "phase2/tiny.f32.gguf").read_bytes()},
        "tokenizer": {"tiny": (fixtures / "phase1/tokenizer.json").read_bytes(),
                      "multilingual": "A\n中文\u200d🙂مرحبا".encode(), "partial-utf8": b"a\xf0\x9f"},
        "quant": {},
    }
    types = json.loads(Path("models/manifest.json").read_text())["reference_checks"]["block_decoder_cross_checks"]
    for phase in ("phase2", "phase3"):
        for kind, fixture in json.loads((fixtures / phase / "blocks.json").read_text()).items():
            for i, packed in enumerate(fixture["blocks"]):
                items["quant"][f"{kind}-{i}"] = bytes([types[kind]["type_id"]]) + bytes.fromhex(packed)
    for name, corpus in items.items():
        for regression in (fixtures / "fuzz" / name).glob("*"):
            if regression.is_file():
                corpus[regression.name] = regression.read_bytes()
    return items


def run(args, name, inputs):
    work = args.work / name
    work.mkdir()
    corpus = work / "corpus"
    corpus.mkdir()
    for seed, data in inputs.items():
        (corpus / seed).write_bytes(data)
    command = [args.build / f"nathr_fuzz_{name}", f"-seed={args.seed}",
               f"-max_total_time={args.seconds}", f"-max_len={1024 if name == 'quant' else 65536}",
               "-print_final_stats=1", "-timeout=5", f"-artifact_prefix={work}/", corpus]
    start = monotonic()
    with (work / "run.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    log_text = (work / "run.log").read_text(errors="replace")
    units = re.search(r"stat::number_of_executed_units:\s*(\d+)", log_text)
    report = {"exit_status": result.returncode, "seed": args.seed, "requested_seconds": args.seconds,
              "wall_seconds": monotonic() - start, "units": int(units[1]) if units else None,
              "seed_sha256": {key: hashlib.sha256(data).hexdigest() for key, data in inputs.items()},
              "findings": sorted(p.name for p in work.iterdir() if p.name not in ("run.log", "corpus"))}
    print(name, report["units"], "units", f"exit {result.returncode}", flush=True)
    return name, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--seed", type=int, default=58053)
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3, 4), default=4)
    if (args := parser.parse_args()).seconds < 1:
        parser.error("--seconds must be positive")
    args.work.mkdir(parents=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run, args, name, data) for name, data in seeds().items()]
        reports = dict(f.result() for f in futures)
    (args.work / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    if any(row["exit_status"] or row["units"] is None for row in reports.values()):
        raise SystemExit("A fuzz target failed. Review the local logs and fixtures.")


if __name__ == "__main__":
    main()
