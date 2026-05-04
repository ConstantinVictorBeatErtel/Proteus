#!/usr/bin/env bash
# Run the three training conditions + evaluate in order (same order as the
# manual nohup chains used on GPU boxes). Uses `set -o pipefail`-safe style
# via running python directly (no `python | tee` masking exit codes here;
# use `tee` from the caller if you want live logs).
#
# Before first run on a machine:
#   - venv + pip install -r requirements.txt
#   - Zarr datasets under data/touch_in_the_wild/four_tasks/...
#   - CLIP cache:  python src/cache_clip.py
#   - Tactile npy cache:  python src/data.py --cache-tactile [--cache-tactile-force]
# Default VTBC_NUM_WORKERS=0 is safest for *.zarr.zip stores; raise if stable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck source=/dev/null
source "${ROOT}/.venv/bin/activate"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export VTBC_NUM_WORKERS="${VTBC_NUM_WORKERS:-0}"

MASTER="${ROOT}/train_pipeline_master.log"
MARKER="${ROOT}/vtbc_training_chain.done"

log_master() {
  printf '%s\n' "$1" >>"${MASTER}"
}

rm -f "${MARKER}"

log_master "[train_chain] start vision_only $(date)"
python3 src/train.py --condition vision_only 2>&1 | tee "${ROOT}/train_vision_only.log"

log_master "[train_chain] start tactile_only $(date)"
python3 src/train.py --condition tactile_only 2>&1 | tee "${ROOT}/train_tactile_only.log"

log_master "[train_chain] start visuo_tactile $(date)"
python3 src/train.py --condition visuo_tactile 2>&1 | tee "${ROOT}/train_visuo_tactile.log"

log_master "[train_chain] start evaluate $(date)"
python3 src/evaluate.py 2>&1 | tee "${ROOT}/evaluate.log"

log_master "[train_chain] ALL DONE $(date)"
touch "${MARKER}"
