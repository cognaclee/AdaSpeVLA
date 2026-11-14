#!/usr/bin/env bash
# One timing run (timing head only) on the given GPUs, so the three timing runs can go in parallel.
# Usage: CUDA_VISIBLE_DEVICES=4 TASK=bread adaspe_train/run_timing_one.sh {20|30|40}
# Same environment and settings as run_train.sh; resumes if a checkpoint exists.
set -euo pipefail

K=$1
TASK=${TASK:-bread}
ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)  # repo root: this script is <root>/adaspe_train/
source "$ROOT/adaspe_train/names.sh"  # TAG, JOB_BASE from VARIANT
RUN=${RUN:-$ROOT/test/franka/$TAG}
PY=$ROOT/external/openpi/.venv/bin/python
mkdir -p "$RUN"
cd "$ROOT/external/openpi"
export TMPDIR=$ROOT/.cache/tmp HF_HUB_OFFLINE=1 HF_LEROBOT_HOME=$ROOT/data/lerobot
export HF_HOME=$ROOT/.cache/huggingface XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export OPENPI_DATA_HOME=$ROOT/.cache/openpi JAX_COMPILATION_CACHE_DIR=$ROOT/.cache/jax
# Keep cluster modules, PYTHONPATH and ~/.local site-packages out of the uv venv.
unset PYTHONPATH LD_LIBRARY_PATH; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
mkdir -p "$TMPDIR"

NGPU=$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}")
NAME=${TAG}_timing_from${K}k
TIM_DIR=$ROOT/checkpoints/adaspe/$NAME/formal
status() { echo "$(date -Is) $*" | tee -a "$RUN/pipeline_status.log"; }
source "$ROOT/adaspe_train/dev_bridge.sh"
stop_dev_chain  # the 2 h dev chain ($JOB_BASE-dev-$TASK) trains the same dirs; a no-op inside the dev link

if [[ -d "$TIM_DIR/9999/params" ]]; then status "timing from ${K}k already done"; exit 0; fi
RESUME=(); [[ -d "$TIM_DIR" ]] && RESUME=(--resume)
status "timing from ${K}k: train to 10000 on GPUs $CUDA_VISIBLE_DEVICES (fsdp=$NGPU) ${RESUME[*]:-}"
$PY "$ROOT/adaspe_train/train.py" "$NAME" --exp_name formal "${RESUME[@]}" --fsdp_devices "$NGPU" >> "$RUN/timing_from${K}k.log" 2>&1
[[ -d "$TIM_DIR/9999/params" ]] || { status "timing from ${K}k FAILED: no step-9999 params"; exit 1; }
status "timing from ${K}k done"
