#!/usr/bin/env bash
# AdaSpeVLA pipeline: preflight (short train + resume) -> formal Action 40k -> 3 timing runs (from 20k/30k/40k).
# Run on already-chosen GPUs (CUDA_VISIBLE_DEVICES); launched by wait_gpus_and_launch.sh.
# Stops at the first failure; formal stages run only after the preflight passes.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)  # repo root: this script is <root>/adaspe_train/
TASK=${TASK:-bread}
RUN=${RUN:-$ROOT/test/franka/${TASK}_adaspe}
PY=$ROOT/external/openpi/.venv/bin/python
TRAIN="$PY $ROOT/adaspe_train/train.py"
mkdir -p "$RUN"
cd "$ROOT/external/openpi"

export TMPDIR=$ROOT/.cache/tmp HF_HUB_OFFLINE=1 HF_LEROBOT_HOME=$ROOT/data/lerobot
export HF_HOME=$ROOT/.cache/huggingface XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export OPENPI_DATA_HOME=$ROOT/.cache/openpi JAX_COMPILATION_CACHE_DIR=$ROOT/.cache/jax
# Keep cluster modules, PYTHONPATH and ~/.local site-packages out of the uv venv.
unset PYTHONPATH LD_LIBRARY_PATH; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
mkdir -p "$TMPDIR"

NGPU=$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:?set by launcher}")
FSDP=(--fsdp_devices "$NGPU")
status() { echo "$(date -Is) $*" | tee -a "$RUN/pipeline_status.log"; }
finite_losses() {  # every logged loss in the log is finite
  ! grep -E "loss=" "$1" | grep -qiE "loss=(nan|inf)" && grep -qE "loss=" "$1"
}

status "start on GPUs $CUDA_VISIBLE_DEVICES (fsdp=$NGPU); git=$(git -C "$ROOT/external/openpi" rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv > "$RUN/gpus_at_start.csv"

# A separate dev preflight (slurm/preflight.slurm, job adaspe-pf-$TASK) may be running: wait so we never share its dir.
if [[ -z "${PREFLIGHT_ONLY:-}" && ! -f "$RUN/preflight_passed" ]]; then
  dev_pf() { squeue -h -u "$USER" -n "adaspe-pf-$TASK" -t RUNNING -o %A 2>/dev/null || true; }
  if [[ -n "$(dev_pf)" ]]; then
    status "waiting for the running dev preflight adaspe-pf-$TASK"
    while [[ -n "$(dev_pf)" ]]; do sleep 60; done
  fi
fi

# 1) Preflight: 30 steps, checkpoint at 15/30, then resume to 40.
if [[ ! -f "$RUN/preflight_passed" ]]; then
  status "preflight: train 30 steps"
  $TRAIN ${TASK}_adaspe_action --exp_name preflight --overwrite \
    --num_train_steps 30 --save_interval 15 --log_interval 5 "${FSDP[@]}" > "$RUN/preflight_train.log" 2>&1
  finite_losses "$RUN/preflight_train.log" || { status "preflight FAILED: non-finite loss"; exit 1; }
  status "preflight: resume to 40"
  $TRAIN ${TASK}_adaspe_action --exp_name preflight --resume \
    --num_train_steps 40 --save_interval 20 --log_interval 5 "${FSDP[@]}" > "$RUN/preflight_resume.log" 2>&1
  finite_losses "$RUN/preflight_resume.log" || { status "preflight FAILED after resume"; exit 1; }
  # A resumed run starts at step 30: it must log step 35 and never step 0.
  { grep -q "Step 35:" "$RUN/preflight_resume.log" && ! grep -q "Step 0:" "$RUN/preflight_resume.log"; } \
    || { status "preflight FAILED: resume did not continue from step 30"; exit 1; }
  touch "$RUN/preflight_passed"
  rm -rf "$ROOT/checkpoints/adaspe/${TASK}_adaspe_action/preflight"  # ~50 GB/ckpt; logs kept
  status "preflight passed"
  [[ -n "${PREFLIGHT_ONLY:-}" ]] && { status "stop after preflight (PREFLIGHT_ONLY)"; exit 0; }
fi

# 2) Formal Action stage to 40k (resumes if a checkpoint exists); 10k/20k/30k/40k are kept.
ACT_DIR=$ROOT/checkpoints/adaspe/${TASK}_adaspe_action/formal
if [[ ! -d "$ACT_DIR/39999/params" ]]; then
  RESUME=(); [[ -d "$ACT_DIR" ]] && RESUME=(--resume)
  status "action: train to 40000 ${RESUME[*]:-}"
  $TRAIN ${TASK}_adaspe_action --exp_name formal "${RESUME[@]}" "${FSDP[@]}" >> "$RUN/action.log" 2>&1
  for s in 20000 30000 39999; do  # OpenPI saves the last step as num_train_steps-1
    [[ -d "$ACT_DIR/$s/params" ]] || { status "action FAILED: no step-$s params"; exit 1; }
  done
fi
status "action done"
[[ -n "${ACTION_ONLY:-}" ]] && { status "stop after action (ACTION_ONLY; timing via run_timing_one.sh)"; exit 0; }

# 3) Timing stage (timing head only), once per action checkpoint: 20k first (matched budget), then 30k, 40k.
for k in 20 30 40; do
  NAME=${TASK}_adaspe_timing_from${k}k
  TIM_DIR=$ROOT/checkpoints/adaspe/$NAME/formal
  if [[ ! -d "$TIM_DIR/9999/params" ]]; then
    RESUME=(); [[ -d "$TIM_DIR" ]] && RESUME=(--resume)
    status "timing from ${k}k: train to 10000 ${RESUME[*]:-}"
    $TRAIN $NAME --exp_name formal "${RESUME[@]}" "${FSDP[@]}" >> "$RUN/timing_from${k}k.log" 2>&1
    [[ -d "$TIM_DIR/9999/params" ]] || { status "timing from ${k}k FAILED: no step-9999 params"; exit 1; }
  fi
done
status "completed"
