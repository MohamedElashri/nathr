"""Measure the fixed Phase 4 workloads with matched repeated batches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time

from phase2_bench import GAIN, NOISE, within_batch


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def measure(command, cpu_set):
    print(f"CPUs {cpu_set} {command[0].name} {' '.join(map(str, command[1:]))}", flush=True)
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = subprocess.run(["taskset", "-c", cpu_set, *command], capture_output=True, text=True, check=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {**json.loads(result.stdout), "wall_s": time.monotonic() - start,
            "cpu_s": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime}


def comparison(batches, baseline, candidate, stages):
    drift = None
    if len(batches) == 2:
        drift = {}
        for name in (baseline, candidate):
            drift[name] = {}
            for stage in stages:
                a, b = [pair[name][f"{stage}_ms"]["median"] for pair in batches]
                drift[name][stage] = abs(b - a) / a
    changes = [{stage: pair[candidate][f"{stage}_tokens_s"] / pair[baseline][f"{stage}_tokens_s"] - 1
                for stage in stages} for pair in batches]
    consistent = all(within_batch(pair[name]) for pair in batches for name in (baseline, candidate))
    stable = (consistent and drift is not None
              and all(delta <= NOISE for stage in drift.values() for delta in stage.values()))
    return {"repeatability_passed": stable, "median_drift": drift, "throughput_change": changes,
            "performance_passed": stable and all(all(delta >= GAIN for delta in x.values()) for x in changes)}


def external(path, model, cpus):
    values = measure([path, "bench", model, str(len(cpus.split(",")))], cpus)
    for stage, column, tokens in (("prefill", 0, 128), ("decode", 1, 32)):
        times = [sample[column] for sample in values["prefill_and_decode_ms"]]
        values[f"{stage}_tokens_s"] = tokens * 1000 / statistics.median(times)
        values[f"{stage}_cv"] = statistics.stdev(times) / statistics.mean(times)
    return values


def topology(cpu_set):
    cpus = [int(cpu) for cpu in cpu_set.split(",")]
    if len(set(cpus)) != len(cpus) or set(cpus) - os.sched_getaffinity(0):
        raise ValueError(f"invalid CPU affinity {cpu_set}")
    core_ids = {}
    for cpu in cpus:
        directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        core_ids[cpu] = [(directory / name).read_text().strip() for name in ("physical_package_id", "core_id")]
    return core_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "binary", "controlled", "llama-probe", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--physical", default="8,10,12,14")
    parser.add_argument("--smt", default="12,13,14,15")
    parser.add_argument("--one-core", default="12")
    parser.add_argument("--artifact", choices=("f32", "Q4_0", "Q8_0"), action="append")
    args = parser.parse_args()
    physical, smt = topology(args.physical), topology(args.smt)
    threads = len(physical)
    if len(set(map(tuple, physical.values()))) != threads or len(smt) != threads:
        parser.error("provide one CPU per physical core and an equally sized SMT affinity")
    one_core = topology(args.one_core)
    if len(one_core) != 1:
        parser.error("provide exactly one CPU for the baseline")
    governor = Path(f"/sys/devices/system/cpu/cpu{args.one_core}/cpufreq/scaling_governor")
    manifest = json.loads(Path("models/manifest.json").read_text())
    report = {"schema_version": 1, "kernel": os.uname().release,
              "baseline_sha256": file_hash(args.baseline), "binary_sha256": file_hash(args.binary),
              "affinity": {"one_core": args.one_core, "physical": args.physical, "smt": args.smt},
              "physical_core_ids": physical, "smt_core_ids": smt,
              "governor": governor.read_text().strip() if governor.exists() else None,
              "gates": {"maximum_cv_and_drift": NOISE, "minimum_gain": GAIN},
              "workload": {"prompt": 128, "decode": 32, "context": 512, "warmups": 2, "runs": 7,
                           "logits": "last per operation", "sampling": False, "kv": "F32"},
              "models": {}}

    def save():
        staged = args.output.with_suffix(".part")
        staged.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staged.replace(args.output)

    for artifact in manifest["controlled"]["files"]:
        recipe = artifact["recipe"]
        if recipe not in (args.artifact or ("f32", "Q4_0", "Q8_0")):
            continue
        model = args.controlled / artifact["filename"]
        digest = file_hash(model)
        if digest != artifact["sha256"]:
            raise ValueError(f"{model}: wrong SHA-256")

        common = ["bench", "--model", model, "--prompt-length", "128", "--decode-length", "32", "--context",
                  "512", "--warmups", "2", "--repetitions", "7", "--json"]
        configs = {
            "phase3_avx2": (args.one_core, [args.baseline, *common, "--backend", "avx2"]),
            "repeated_gemv": (args.physical, [args.binary, *common, "--backend", "avx2", "--threads",
                                             str(threads), "--prefill-chunk", "1"]),
            "blocked": (args.physical, [args.binary, *common, "--backend", "avx2", "--threads",
                                       str(threads), "--prefill-chunk", "16"]),
        }
        batches = []
        result = {"sha256": digest, "batches": batches}
        report["models"][recipe] = result
        for order in (tuple(configs), tuple(reversed(configs))):
            batch = {}
            batches.append(batch)
            for name in order:
                affinity, command = configs[name]
                batch[name] = measure(command, affinity)
                save()
            if len(batches) == 1 and not all(map(within_batch, batch.values())):
                result["stopped_after_first_batch"] = "within-batch variability exceeds the 5% limit"
                print(f"{recipe}: first batch exceeded 5% variability", flush=True)
                break
        result["blocked_vs_phase3_avx2"] = comparison(batches, "phase3_avx2", "blocked", ("prefill", "decode"))
        result["blocked_vs_repeated_prefill"] = comparison(batches, "repeated_gemv", "blocked", ("prefill",))
        result["llama_cpp"] = external(args.llama_probe, model, args.physical)
        if recipe == "Q8_0":
            result["smt"] = measure(configs["blocked"][1], args.smt)
        save()


if __name__ == "__main__":
    main()
