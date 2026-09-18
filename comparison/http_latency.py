#!/usr/bin/env python3
"""Measure HTTP search latency at concurrency 1 and 10."""
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--payload", required=True)
parser.add_argument("--requests", type=int, default=100)
parser.add_argument("--warmup-requests", type=int, default=0)
args = parser.parse_args()
payload = args.payload.encode()

def request(_):
    started = time.perf_counter()
    req = urllib.request.Request(args.url, data=payload,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
        response.read()
    return (time.perf_counter() - started) * 1000

def run_requests(concurrency, count, measure):
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        values = list(pool.map(request, range(count)))
    return values if measure else None

for concurrency in (1, 10):
    run_requests(concurrency, args.warmup_requests, measure=False)
    batch_started = time.perf_counter()
    values = sorted(run_requests(concurrency, args.requests, measure=True))
    elapsed = time.perf_counter() - batch_started
    percentile = lambda p: values[min(len(values) - 1, int(len(values) * p / 100))]
    print(json.dumps({
        "concurrency": concurrency,
        "requests": args.requests,
        "p50_ms": round(percentile(50), 3),
        "p90_ms": round(percentile(90), 3),
        "p99_ms": round(percentile(99), 3),
        "mean_ms": round(statistics.mean(values), 3),
        "throughput_per_s": round(args.requests / elapsed, 2),
    }))
