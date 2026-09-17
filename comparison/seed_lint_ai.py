#!/usr/bin/env python3
"""Seed a Lint-AI server with one bulk /add request."""
import argparse
import json
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8080/add")
parser.add_argument("--count", type=int, default=23366)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--bulk", action="store_true",
                    help="send all bounded add requests through /add/batch")
args = parser.parse_args()
if args.batch_size < 1 or args.batch_size > 1024:
    raise SystemExit("--batch-size must be between 1 and 1024")
requests = []
for batch_start in range(0, args.count, args.batch_size):
    batch_end = min(args.count, batch_start + args.batch_size)
    messages = [{"role": "user", "content":
                 f"user memory record {i}: deployment configuration and system decision {i}"}
                for i in range(batch_start, batch_end)]
    request = {"request_id": f"comparison-seed-{batch_start}",
                       "messages": messages, "user_id": "bench-user",
                       "session_id": "bench-session"}
    if args.bulk:
        requests.append(request)
        continue
    body = json.dumps(request).encode()
    req = urllib.request.Request(args.url, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as response:
        if response.status != 200:
            raise SystemExit(f"seed batch failed: HTTP {response.status}")
if args.bulk:
    base_url = args.url[:-4] if args.url.endswith("/add") else args.url.rstrip("/")
    for offset in range(0, len(requests), 8):
        body = json.dumps(requests[offset:offset + 8]).encode()
        req = urllib.request.Request(base_url + "/add/batch", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as response:
            if response.status != 200:
                raise SystemExit(f"bulk seed failed: HTTP {response.status}")
print(args.count)
