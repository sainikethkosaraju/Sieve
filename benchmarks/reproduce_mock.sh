#!/bin/sh
# Offline, synthetic, no-network end-to-end run of every evaluation stage on the fake backend.
# Everything it prints and writes is SYNTHETIC: it tests the pipeline, not any model.
# Usage: benchmarks/reproduce_mock.sh [WORKDIR] [REPORT_DIR]
set -eu
PY="${PY:-$(if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)}"
WORK="${1:-${TMPDIR:-/tmp}/sieve-mock}"
OUT="${2:-results/synthetic-demo}"
rm -rf "$WORK"
"$PY" -m sieve.eval mock --workdir "$WORK" --out "$OUT"
echo "report: $OUT/index.html"
