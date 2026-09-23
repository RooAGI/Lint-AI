#!/bin/bash
# Pillar 1: N independent runs of the LongMemEval-S scoped haystack benchmark.
# Run 1 is wrapped in run_with_peak_rss.py for peak-RSS capture (pillar 2).
#
# Env overrides: BIN, DATA, OUTDIR, RUNS
#   BIN    - haystack_scoped_benchmark binary (default: <repo>/target/release/...)
#   DATA   - LongMemEval-S cleaned JSON (default: <repo>/benchmark/data/longmemeval_s_cleaned.json)
#   OUTDIR - where run logs/results go (default: <repo>/benchmark/results/strong_eval)
#   RUNS   - number of independent runs (default: 5)
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BIN:-$ROOT/target/release/haystack_scoped_benchmark}"
DATA="${DATA:-$ROOT/benchmark/data/longmemeval_s_cleaned.json}"
OUTDIR="${OUTDIR:-$ROOT/benchmark/results/strong_eval}"
RUNS="${RUNS:-5}"
mkdir -p "$OUTDIR"
i=1
while [ "$i" -le "$RUNS" ]; do
  out="$OUTDIR/haystack_run${i}.json"
  log="$OUTDIR/haystack_run${i}.log"
  if [ "$i" -eq 1 ]; then
    python3 "$ROOT/benchmark/strong_eval/run_with_peak_rss.py" "$log" \
      "$BIN" --longmemeval "$DATA" \
      --k 1 --k 3 --k 5 --k 10 --k 20 \
      --out "$out" >> "$log" 2>&1
  else
    "$BIN" --longmemeval "$DATA" \
      --k 1 --k 3 --k 5 --k 10 --k 20 \
      --out "$out" > "$log" 2>&1
  fi
  echo "RUN${i}_EXIT:$?"
  i=$((i + 1))
done
