"""Record release-candidate load, first-logit latency, sustained decode and memory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from tempfile import TemporaryDirectory

from models import check
from phase4_bench import measure, topology


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "native", "controlled", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cpus", default="8,10,12,14")
    parser.add_argument("--warmups", type=int, default=1, choices=range(0, 101))
    parser.add_argument("--repetitions", type=int, default=3, choices=range(1, 101))
    args = parser.parse_args()
    cores = topology(args.cpus)
    if len(set(map(tuple, cores.values()))) != len(cores):
        parser.error("choose one logical CPU per physical core")

    manifest = json.loads(Path("models/manifest.json").read_text())
    native_source = next(s for s in manifest["sources"] if s["repo"] == manifest["canonical_model"])
    for item in native_source["files"]:
        if item["required"]:
            check(args.native / item["filename"], item)
    controlled = {item["recipe"]: item for item in manifest["controlled"]["files"]}
    models = {"native": (args.native, digest(args.native / "model.safetensors"))}
    for recipe in ("Q8_0", "Q4_0"):
        item = controlled[recipe]
        model = args.controlled / item["filename"]
        models[recipe] = (model, check(model, item))

    binary = args.binary.resolve()
    dependencies = subprocess.check_output(["readelf", "-d", binary], text=True)
    with TemporaryDirectory(prefix="nathr-size-") as directory:
        stripped = Path(directory) / binary.name
        shutil.copyfile(binary, stripped)
        subprocess.run(["strip", "--strip-all", stripped], check=True)
        stripped_bytes = stripped.stat().st_size

    governor = Path(f"/sys/devices/system/cpu/cpu{next(iter(cores))}/cpufreq/scaling_governor")
    cpu_info = Path("/proc/cpuinfo").read_text().split("\n\n")[0].splitlines()
    report = {"schema_version": 1, "binary_sha256": digest(binary),
              "manifest_sha256": digest(Path("models/manifest.json")),
              "kernel": platform.release(), "architecture": platform.machine(),
              "cpu": next((line.split(":", 1)[1].strip() for line in cpu_info if line.startswith("model name")), None),
              "physical_core_ids": cores, "observed_load_average": os.getloadavg(),
              "governor": governor.read_text().strip() if governor.exists() else None,
              "binary_bytes": binary.stat().st_size, "stripped_binary_bytes": stripped_bytes,
              "dynamic_dependencies": [line.split("[", 1)[1].split("]", 1)[0]
                                       for line in dependencies.splitlines() if "(NEEDED)" in line],
              "workload": {"prompt": 128, "decode": 128, "context": 512, "warmups": args.warmups,
                           "repetitions": args.repetitions, "kv": "F32", "threads": len(cores),
                           "sampling": False, "ids": "17 + (i * 7919 + 12345) % (49152 - 17)"},
              "scope": "release characterization; the matched improvement gates use reference/phase4_bench.py",
              "models": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, (model, sha) in models.items():
        result = measure([binary, "bench", "--model", model, "--backend", "auto", "--threads", str(len(cores)),
                          "--context", "512", "--prompt-length", "128", "--decode-length", "128",
                          "--warmups", str(args.warmups), "--repetitions", str(args.repetitions), "--json"], args.cpus)
        report["models"][name] = {"weight_sha256": sha, **result}
        partial = args.output.with_suffix(".part")
        partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        partial.replace(args.output)


if __name__ == "__main__":
    main()
