"""Measure concurrent generation through the local streaming API."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection, HTTPException
import json
import math
import random
from threading import Lock
import time
from urllib.parse import urlencode


def distribution(samples):
    if not samples:
        return None
    ordered = sorted(samples)
    count = len(ordered)
    return {"p50": (ordered[(count - 1) // 2] + ordered[count // 2]) / 2,
            "p95": ordered[math.ceil(0.95 * count) - 1]}


def gap_distribution(histogram):
    count = sum(histogram.values())
    if not count:
        return None
    ranks = [(count + 1) // 2, count // 2 + 1, math.ceil(0.95 * count)]
    values = []
    cumulative = 0
    for bucket, frequency in sorted(histogram.items()):
        cumulative += frequency
        while len(values) < len(ranks) and cumulative >= ranks[len(values)]:
            values.append(2 ** ((bucket + 0.5) / 64))
    return {"p50": (values[0] + values[1]) / 2, "p95": values[2]}


def benchmark(args):
    parser = argparse.ArgumentParser(prog="nathr bench --serve", description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--port", type=int, default=8000, help="port of the running local server")
    parser.add_argument("--model", help="API model ID, otherwise the server default")
    parser.add_argument("--requests", type=int, default=16, help="measured requests")
    parser.add_argument("--concurrency", type=int, default=4, help="requests in flight")
    parser.add_argument("--warmups", type=int, default=0, help="unmeasured requests before the trial")
    parser.add_argument("--workload", choices=("unique", "shared-prefix", "mixed"), default="mixed")
    parser.add_argument("--prompt-length", type=int, default=128, help="long prompt length in tokens")
    parser.add_argument("--decode-length", type=int, default=16, help="maximum generated tokens per request")
    parser.add_argument("--seed", type=int, help="reproduce the synthetic prompts")
    parser.add_argument("--timeout", type=float, default=120, help="socket timeout in seconds")
    options = parser.parse_args(args)
    if not (1 <= options.port <= 65535 and 1 <= options.concurrency <= options.requests <= 10000 and
            options.concurrency <= 128 and 0 <= options.warmups <= 100 and
            0 < options.timeout <= 600 and options.prompt_length > 0 and options.decode_length > 0):
        parser.error("invalid request counts, lengths, port, or timeout")
    if options.workload == "shared-prefix" and options.prompt_length < 32:
        parser.error("shared-prefix workloads need at least 32 prompt tokens")

    query = "?" + urlencode({"model": options.model}) if options.model else ""

    def get(path):
        connection = HTTPConnection("127.0.0.1", options.port, timeout=options.timeout)
        try:
            connection.request("GET", path + query)
            reply = connection.getresponse()
            if reply.status != 200:
                raise ValueError(f"{path} returned HTTP {reply.status}")
            return json.load(reply)
        finally:
            connection.close()

    try:
        model = get("/model")
        query = "?" + urlencode({"model": model["id"]})
        required_context = (max(options.prompt_length, options.decode_length) if model["architecture"] == "t5"
                            else options.prompt_length + options.decode_length - 1)
        if (not model["generation"] or required_context > model["generation_context"] or
                options.decode_length > model["limits"]["max_output_tokens"]):
            parser.error("workload exceeds the checkpoint's generation limits")
        seed = options.seed if options.seed is not None else random.randrange(1 << 63)
        vocabulary = model["vocabulary"]
        first_id = min(17, vocabulary // 2)

        def tokens(count, rng):
            return [rng.randrange(first_id, vocabulary) for _ in range(count)]

        prefix = (tokens(options.prompt_length // 2, random.Random(seed))
                  if options.workload == "shared-prefix" else [])

        def sample(index, gaps):
            length, maximum = options.prompt_length, options.decode_length
            short = options.workload == "mixed" and index % 4 != 0
            if short:
                length, maximum = max(1, length // 8), max(1, maximum // 4)
            prompt = prefix + tokens(length - len(prefix), random.Random(f"{seed}:{index}"))
            body = json.dumps({"model": model["id"], "tokens": prompt, "max_new_tokens": maximum,
                               "greedy": True, "stream": True}, separators=(",", ":"))
            connection = HTTPConnection("127.0.0.1", options.port, timeout=options.timeout)
            start = time.monotonic()
            first = last = None
            pending_gaps = Counter()
            done = None
            try:
                connection.request("POST", "/infer", body, {"Content-Type": "application/json"})
                reply = connection.getresponse()
                if reply.status != 200:
                    detail = reply.read(4096).decode(errors="replace").strip()
                    return {"error": f"HTTP {reply.status} {detail}".strip()}
                for line in reply:
                    if not line.startswith(b"data: "):
                        continue
                    if line.rstrip() == b"data: [DONE]":
                        if done is None:
                            return {"error": "stream ended before generation finished"}
                        gaps.update(pending_gaps)
                        return {"ttft_ms": 1000 * (first - start) if first is not None else None, "short": short,
                                "queue_ms": done.get("queue_ms"), "output_tokens": done["emitted"],
                                "cached_prompt_tokens": done.get("cached_prompt_tokens", 0)}
                    event = json.loads(line[6:])
                    if "error" in event:
                        return {"error": event["error"]["message"]}
                    if event["event"] == "token":
                        now = time.monotonic()
                        if last is None:
                            first = now
                        else:
                            # Bound storage across long trials with logarithmic bins of about 1%.
                            gap_ms = max(0.001, 1000 * (now - last))
                            pending_gaps[math.floor(64 * math.log2(gap_ms))] += 1
                        last = now
                    elif event["event"] == "done":
                        done = event
                return {"error": "incomplete stream"}
            except (OSError, HTTPException, ValueError) as error:
                return {"error": str(error)}
            finally:
                connection.close()

        indices = iter(range(options.requests))
        lock = Lock()

        def lane(_):
            samples, gaps = [], Counter()
            while True:
                with lock:
                    index = next(indices, None)
                if index is None:
                    return samples, gaps
                samples.append(sample(index, gaps))

        for index in range(options.warmups):
            warmup = sample(-2 - index, Counter())
            if "error" in warmup:
                parser.error(f"warmup failed: {warmup['error']}")
        before = get("/health")
        with ThreadPoolExecutor(max_workers=options.concurrency) as executor:
            start = time.monotonic()
            lanes = list(executor.map(lane, range(options.concurrency)))
            elapsed = time.monotonic() - start
        after = get("/health")
    except (KeyError, TypeError) as error:
        parser.error(f"unexpected response from the Nathr server: {error}")
    except (OSError, HTTPException, ValueError) as error:
        parser.error(str(error))

    samples, gaps = [], Counter()
    for results, lane_gaps in lanes:
        samples.extend(results)
        gaps.update(lane_gaps)
    errors = [sample["error"] for sample in samples if "error" in sample]
    completed = [sample for sample in samples if "error" not in sample]
    output_tokens = sum(sample["output_tokens"] for sample in completed)
    report = {
        "model": model["id"], "workload": options.workload, "seed": seed,
        "concurrency": options.concurrency, "requests": options.requests, "failed": len(errors),
        "errors": sorted(set(errors)), "prompt_length": options.prompt_length, "decode_length": options.decode_length,
        "elapsed_s": elapsed, "requests_per_s": len(completed) / elapsed, "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / elapsed,
        "ttft_ms": distribution([s["ttft_ms"] for s in completed if s["ttft_ms"] is not None]),
        "inter_token_ms": gap_distribution(gaps),
        "queue_ms": distribution([s["queue_ms"] for s in completed if s["queue_ms"] is not None]),
        "cached_prompt_tokens": sum(s["cached_prompt_tokens"] for s in completed),
        "kv_peak_before_bytes": before["kv"]["peak_bytes"], "kv_peak_bytes": after["kv"]["peak_bytes"],
        "queue_bypasses": after.get("queue_bypasses", 0) - before.get("queue_bypasses", 0),
    }
    if options.workload == "mixed":
        for name, short in (("short", True), ("long", False)):
            group = [s for s in completed if s["short"] == short]
            report[name] = {"requests": len(group),
                            "ttft_ms": distribution([s["ttft_ms"] for s in group if s["ttft_ms"] is not None]),
                            "queue_ms": distribution([s["queue_ms"] for s in group if s["queue_ms"] is not None])}
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)
