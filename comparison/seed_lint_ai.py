#!/usr/bin/env python3
"""Seed a Lint-AI server with one bulk /add request."""
import argparse
import json
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8080/add")
parser.add_argument("--count", type=int, default=23366)
args = parser.parse_args()
messages = [{"role": "user", "content":
             f"user memory record {i}: deployment configuration and system decision {i}"}
            for i in range(args.count)]
body = json.dumps({"request_id": "comparison-seed", "messages": messages,
                   "user_id": "bench-user", "session_id": "bench-session"}).encode()
req = urllib.request.Request(args.url, data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=300) as response:
    print(response.status)
