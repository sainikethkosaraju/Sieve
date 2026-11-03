#!/usr/bin/env bash
# Push a script to Kaggle as a private GPU kernel, poll, and fetch its log.
# Auth: the Kaggle bearer token is injected by the environment's network secret; no local credential.
#
# Usage:
#   KAGGLE_USER=<kaggle-username> [KERNEL_DATA_SOURCES="user/kernel-slug user/other"] \
#     kaggle/kaggle_run.sh <script.py> <slug> [outdir]
#
# KAGGLE_USER          required: the Kaggle username that owns the kernel
# KERNEL_DATA_SOURCES  optional, space/comma separated "owner/kernel-slug" list; each kernel's
#                      output is mounted under /kaggle/input/<kernel-slug>/ (cached llama-server,
#                      or the previous run's workdir for resuming)
# DATASET_DATA_SOURCES optional, same for datasets
# ENABLE_GPU           1 (default) or 0
# MAX_WAIT_MIN         give up polling after this many minutes (default 120; the kernel keeps running)
# outdir               default kaggle_out/<slug>-<UTC timestamp> (one directory per run, never reused)
#
# Kaggle's output *files* are served from kaggleusercontent.com, which the Sieve sandbox cannot
# reach: downloads are attempted and any failure is only logged. Results travel in the kernel log
# (see kaggle/decode_log_bundle.py).
set -euo pipefail
: "${KAGGLE_USER:?set KAGGLE_USER to the Kaggle username that owns the kernel}"
SCRIPT="$1"; SLUG="$2"
OUT="${3:-kaggle_out/$SLUG-$(date -u +%Y%m%dT%H%M%SZ)}"
API="https://www.kaggle.com/api/v1"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-120}"
mkdir -p "$OUT"
export SCRIPT SLUG OUT KAGGLE_USER ENABLE_GPU="${ENABLE_GPU:-1}" \
  KERNEL_DATA_SOURCES="${KERNEL_DATA_SOURCES:-}" DATASET_DATA_SOURCES="${DATASET_DATA_SOURCES:-}"
python3 -I - <<'EOF'
import json, os, re
def split(v):
    return [x for x in re.split(r"[\s,]+", v) if x]
script, slug, out = os.environ["SCRIPT"], os.environ["SLUG"], os.environ["OUT"]
body = {"slug": f"{os.environ['KAGGLE_USER']}/{slug}", "newTitle": slug, "text": open(script).read(),
        "language": "python", "kernelType": "script", "isPrivate": True,
        "enableGpu": os.environ["ENABLE_GPU"] == "1", "enableTpu": False, "enableInternet": True,
        "datasetDataSources": split(os.environ["DATASET_DATA_SOURCES"]), "competitionDataSources": [],
        "kernelDataSources": split(os.environ["KERNEL_DATA_SOURCES"]), "modelDataSources": [],
        "categoryIds": []}
json.dump(body, open(out + "/push.json", "w"))
print("kernel data sources:", body["kernelDataSources"], "| dataset data sources:", body["datasetDataSources"])
EOF
curl -sS -m 120 -X POST "$API/kernels/push" -H "Content-Type: application/json" --data @"$OUT/push.json" \
  | tee "$OUT/push_response.json" | cut -c1-400; echo
if grep -q '"error"' "$OUT/push_response.json" 2>/dev/null && ! grep -q '"ref"' "$OUT/push_response.json"; then
  echo "push failed" >&2; exit 1
fi
START=$(date +%s)
while true; do
  S=$(curl -sS -m 30 "$API/kernels/status?userName=$KAGGLE_USER&kernelSlug=$SLUG" || true)
  echo "$(date +%T) $S" | cut -c1-200
  case "$S" in *'"complete"'*|*'"error"'*|*'"cancel'*) break;; esac
  if [ $(( ($(date +%s) - START) / 60 )) -ge "$MAX_WAIT_MIN" ]; then echo "MAX_WAIT_MIN reached; kernel still running" >&2; break; fi
  sleep 30
done
echo "$S" > "$OUT/status.json"
curl -sS -m 180 "$API/kernels/output?userName=$KAGGLE_USER&kernelSlug=$SLUG" -o "$OUT/output.json"
python3 -I - "$OUT/output.json" "$OUT" <<'EOF'
import json, sys, urllib.request
d = json.load(open(sys.argv[1]))
out = sys.argv[2]
try:
    log = "".join(e.get("data", "") for e in json.loads(d.get("log", "[]")))
except Exception:
    log = d.get("log", "")
open(out + "/log.txt", "w").write(log)
print("log bytes:", len(log), "| hasNextPageToken:", d.get("hasNextPageToken"))
for f in d.get("files", []):
    name = f.get("fileName", "file")
    try:
        urllib.request.urlretrieve(f["url"], out + "/" + name.replace("/", "_"))
        print("fetched", name)
    except Exception as e:  # expected from the sandbox (kaggleusercontent.com blocked): log only
        print("could not fetch", name, "(log only):", e)
EOF
echo "output in $OUT"
