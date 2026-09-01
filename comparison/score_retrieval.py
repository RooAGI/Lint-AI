#!/usr/bin/env python3
"""Run the canonical shared retrieval scorer from the comparison directory."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent.parent / "benchmark" / "score_retrieval.py"), run_name="__main__")
