"""Run the matched Phase 2 benchmark and report the predeclared performance gates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time


NOISE = 0.05
GAIN = 0.10


def measure(command, cpu):
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = subprocess.run(["taskset", "-c", str(cpu), *command], text=True, capture_output=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed: {result.stderr.strip()}")
    samples = json.loads(result.stdout)
    samples["process_cpu_time"] = {"wall_s": time.monotonic() - start,
                                   "user_s": after.ru_utime - before.ru_utime,
                                   "system_s": after.ru_stime - before.ru_stime}
    return samples


def within_batch(batch):
    return all(batch[f"{stage}_ms"]["coefficient_of_variation"] <= NOISE for stage in ("prefill", "decode"))


def gate(pairs):
    answer = {"repeatability_passed": False, "performance_passed": False}
    if len(pairs) != 2 or not all(within_batch(batch) for pair in pairs for batch in pair.values()):
        return answer
    drift = {}
    for policy in ("scalar", "avx2"):
        drift[policy] = {}
        for stage in ("prefill", "decode"):
            a, b = [pair[policy][f"{stage}_ms"]["median"] for pair in pairs]
            drift[policy][stage] = abs(b - a) / a
    answer["median_drift"] = drift
    answer["repeatability_passed"] = all(value <= NOISE for pair in drift.values() for value in pair.values())
    if answer["repeatability_passed"]:
        changes = [{stage: pair["avx2"][f"{stage}_tokens_s"] / pair["scalar"][f"{stage}_tokens_s"] - 1
                    for stage in ("prefill", "decode")} for pair in pairs]
        answer["throughput_change"] = changes
        answer["performance_passed"] = all(pair["prefill"] >= -NOISE and pair["decode"] >= GAIN for pair in changes)
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    host = os.uname()
    cpufreq = Path(f"/sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_governor")
    cpus = Path("/proc/cpuinfo").read_text().split("\n\n")
    cpu_records = [dict(line.split(":", 1) for line in cpu.splitlines() if ":" in line) for cpu in cpus]
    cpu_records = [{key.strip(): value.strip() for key, value in cpu.items()} for cpu in cpu_records]
    cpu_name = next((cpu.get("model name") for cpu in cpu_records if cpu.get("processor") == str(args.cpu)), None)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip()
    with args.binary.open("rb") as stream:
        binary_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "schema_version": 1, "host": {"kernel": host.release, "architecture": host.machine, "affinity": [args.cpu],
                                      "cpu": cpu_name,
                                      "observed_governor": cpufreq.read_text().strip() if cpufreq.is_file() else None},
        "workload": {"prompt": 128, "decode": 32, "context": 512, "warmups": 2, "repetitions": 7,
                     "kv_dtype": "F32", "logits": "last token per operation", "sampling": False,
                     "ids": "17 + (i * 7919 + 12345) % (49152 - 17)"},
        "gates": {"maximum_cv_and_median_drift": NOISE, "minimum_decode_gain": GAIN, "maximum_prefill_loss": NOISE},
        "models": {}, "git_revision": revision or None, "binary_sha256": binary_hash,
    }

    def save():
        temporary = args.output.with_suffix(".part")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    artifacts = [item for item in manifest["controlled"]["files"] if item["recipe"] in ("Q8_0", "Q4_0")]
    for item in artifacts:
        path = args.controlled / item["filename"]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError(f"{path}: wrong SHA-256")
        result = {"sha256": item["sha256"], "batches": []}
        report["models"][item["recipe"]] = result
        for order in (("scalar", "avx2"), ("avx2", "scalar")):
            pair = {}
            for backend in order:
                print(f"{item['recipe']} pair {len(result['batches']) + 1}: {backend}", flush=True)
                pair[backend] = measure([args.binary, "bench", "--model", path, "--backend", backend, "--threads", "1",
                                         "--prompt-length", "128", "--decode-length", "32", "--context", "512",
                                         "--warmups", "2", "--repetitions", "7", "--json"], args.cpu)
            result["batches"].append(pair)
            result.update(gate(result["batches"]))
            save()
            if not all(within_batch(batch) for batch in pair.values()):
                print(f"{item['recipe']}: within-batch repeatability failed", flush=True)
                break
    for item in artifacts:
        path = args.controlled / item["filename"]
        report["models"][item["recipe"]]["components"] = measure([args.probes, "--gguf-probes", path], args.cpu)
        save()


if __name__ == "__main__":
    main()
