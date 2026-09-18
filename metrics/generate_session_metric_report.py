#!/usr/bin/env python3
"""Stable CLI entry point for generating a session metric report."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


IMPLEMENTATION = (
    Path(__file__).resolve().parent.parent
    / "metrics-sess-019fe4c9-abf8-71e1-b086-b5eb323f3790"
    / "generate_session_metric_report.py"
)


def main() -> int:
    if not IMPLEMENTATION.exists():
        raise SystemExit(f"metric implementation not found: {IMPLEMENTATION}")
    spec = importlib.util.spec_from_file_location("lint_ai_session_metrics", IMPLEMENTATION)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load metric implementation: {IMPLEMENTATION}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
