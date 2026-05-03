#!/usr/bin/env bash
# Full VT-BC pipeline (run from repo root with venv activated).
set -euo pipefail
cd "$(dirname "$0")"
: "${VTBC_NUM_WORKERS:=4}"
export VTBC_NUM_WORKERS
echo "[pipeline] nvidia-smi driver:" && nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo "[pipeline] Step 1/4: cache_clip"
python src/cache_clip.py
echo "[pipeline] Step 2/4: train all conditions"
python src/train.py --condition all
echo "[pipeline] Step 3/4: evaluate"
python src/evaluate.py
echo "[pipeline] Step 4/4: figures"
python notebooks/02_results.py
echo "[pipeline] DONE"
