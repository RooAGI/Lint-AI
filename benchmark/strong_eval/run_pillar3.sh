#!/bin/bash
# Pillar 3: write-path scaling. RUNS independent refresh-scaling runs per corpus
# size. Each binary invocation reports cold_refresh_ms + median-of-3
# single_write_refresh_ms.
# The 10k batch is skipped when available RAM is under 2GB (a prior 10k run
# left only ~317MB free on a 7.9GB box).
#
# Env overrides: BIN, DATA, OUTDIR, SIZES, RUNS
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BIN:-$ROOT/target/release/refresh_scaling_benchmark}"
DATA="${DATA:-$ROOT/benchmark/data/longmemeval_s_cleaned.json}"
OUTDIR="${OUTDIR:-$ROOT/benchmark/results/strong_eval}"
SIZES="${SIZES:-1000 2500 5000 10000}"
RUNS="${RUNS:-5}"
mkdir -p "$OUTDIR"
for size in $SIZES; do
  if [ "$size" -eq 10000 ]; then
    avail_mb=$(free -m | awk '/^Mem:/{print $7}')
    echo "10k batch: ${avail_mb}MB available"
    if [ "$avail_mb" -lt 2048 ]; then
      echo "SIZE10000_SKIPPED: only ${avail_mb}MB available (< 2048MB guard)"
      continue
    fi
  fi
  i=1
  while [ "$i" -le "$RUNS" ]; do
    "$BIN" --sizes "$size" --longmemeval "$DATA" \
      > "$OUTDIR/refresh_${size}_run${i}.log" 2>&1
    echo "SIZE${size}_RUN${i}_EXIT:$?"
    i=$((i + 1))
  done
done
