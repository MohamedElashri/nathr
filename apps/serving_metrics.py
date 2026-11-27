"""Bounded per-model serving statistics and Prometheus text exposition."""

from bisect import bisect_left
from itertools import accumulate


GAUGES = {
    "worker_up": "Whether the model worker is available.",
    "requests_pending": "Requests in the frontend's worker pending table.",
    "requests_active": "Active generation requests at the last scheduler boundary.",
    "requests_queued": "Queued generation requests at the last scheduler boundary.",
    "queued_tokens": "Prompt tokens awaiting admission to the generation scheduler.",
    "kv_used_bytes": "KV page bytes in use, including retained prefix pages.",
    "kv_resident_bytes": "KV page bytes held by the page pool, including free pages.",
    "kv_budget_bytes": "Configured KV page budget in bytes.",
}

COUNTERS = {
    "requests_finished_total": "Native requests with a terminal event.",
    "requests_cancelled_total": "Requests cancelled before or during inference.",
    "generated_tokens_total": "Generation tokens emitted by the native worker.",
    "prefix_cache_lookups_total": "Generation requests eligible for prefix reuse.",
    "prefix_cache_hits_total": "Generation requests that reused at least one KV page.",
    "prefix_cache_tokens_total": "Prompt tokens reused through the prefix cache.",
}

HISTOGRAMS = {
    "request_duration_seconds": "Time from frontend submission to a native terminal event.",
    "request_queue_time_seconds": "Time spent waiting in the native generation queue.",
    "time_to_first_token_seconds": "Time from native submission to the first generated token.",
    "inter_token_latency_seconds": "Time between successive generated tokens.",
}

BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class Histogram:
    def __init__(self):
        self.counts = [0] * (len(BUCKETS) + 1)
        self.total = 0.0

    def observe(self, seconds):
        self.counts[bisect_left(BUCKETS, seconds)] += 1
        self.total += seconds

    def snapshot(self):
        return tuple(self.counts), self.total


class ModelMetrics:
    def __init__(self):
        self.scheduler = {}
        self.finished = 0
        self.generated = 0
        self.histograms = {name: Histogram() for name in HISTOGRAMS}

    def observe(self, event, duration=None):
        if duration is not None:
            self.finished += 1
            self.histograms["request_duration_seconds"].observe(duration)
            if "queue_ms" in event:
                self.histograms["request_queue_time_seconds"].observe(event["queue_ms"] / 1000)
        elif event["event"] == "token":
            self.generated += 1
            if event["emitted"] == 1:
                self.histograms["time_to_first_token_seconds"].observe(event["elapsed_ms"] / 1000)
            if "inter_token_ms" in event:
                self.histograms["inter_token_latency_seconds"].observe(event["inter_token_ms"] / 1000)

    def snapshot(self, *, up, pending, cancelled):
        values = dict(self.scheduler, worker_up=int(up), requests_pending=pending,
                      requests_cancelled_total=cancelled, requests_finished_total=self.finished,
                      generated_tokens_total=self.generated)
        if not up:
            for name in ("requests_active", "requests_queued", "queued_tokens", "kv_used_bytes", "kv_resident_bytes"):
                values[name] = 0
        return values, {name: histogram.snapshot() for name, histogram in self.histograms.items()}


def render_metrics(snapshots):
    models = []
    for model, (values, histograms) in snapshots.items():
        escaped = model.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
        models.append((f'model="{escaped}"', values, histograms))
    lines = []
    for kind, families in (("gauge", GAUGES), ("counter", COUNTERS), ("histogram", HISTOGRAMS)):
        for name, description in families.items():
            metric = f"nathr_{name}"
            lines.extend((f"# HELP {metric} {description}", f"# TYPE {metric} {kind}"))
            for label, values, histograms in models:
                if kind != "histogram":
                    lines.append(f"{metric}{{{label}}} {values.get(name, 0)}")
                    continue
                counts, total = histograms[name]
                cumulative = list(accumulate(counts))
                for upper, count in zip((*BUCKETS, "+Inf"), cumulative):
                    lines.append(f'{metric}_bucket{{{label},le="{upper}"}} {count}')
                lines.extend((f"{metric}_count{{{label}}} {cumulative[-1]}", f"{metric}_sum{{{label}}} {total}"))
    return ("\n".join(lines) + "\n").encode()
