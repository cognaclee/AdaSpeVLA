#!/usr/bin/bash
#SBATCH --job-name=adaspe-tlr
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --hint=nomultithread
#SBATCH --exclude=jzxh008,jzxh043,jzxh045,jzxh018
#SBATCH --output=test/franka/%x-%j.out
# One higher-LR timing run (configs.TIMINGS_LR, 10k steps, from Action 39999) on 1 GPU; resumes if a checkpoint exists.
#   sbatch -A yfl@h100 --job-name=adaspe-tlr-<NAME> --export=ALL,TASK=insert_tube,NAME=insert_tube_adaspe_expert_timing_from40k_lr1e-3 \
#     adaspe_train/slurm/timing_lr.slurm
set -euo pipefail
unset SLURM_CPU_BIND
: "${TASK:?}" "${NAME:?}"
ROOT=$SLURM_SUBMIT_DIR
TAG=${NAME%%_timing_*}
RUN=$ROOT/test/franka/$TAG
DIR=$ROOT/checkpoints/adaspe/$NAME/formal
mkdir -p "$RUN"
cd "$ROOT/external/openpi"
export TMPDIR=$ROOT/.cache/tmp HF_HUB_OFFLINE=1 HF_LEROBOT_HOME=$ROOT/data/lerobot
export HF_HOME=$ROOT/.cache/huggingface XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export OPENPI_DATA_HOME=$ROOT/.cache/openpi JAX_COMPILATION_CACHE_DIR=$ROOT/.cache/jax
unset PYTHONPATH LD_LIBRARY_PATH; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
status() { echo "$(date -Is) $*" | tee -a "$RUN/pipeline_status.log"; }
if [[ -d "$DIR/9999/params" ]]; then status "$NAME already done"; exit 0; fi
RESUME=(); [[ -d "$DIR" ]] && RESUME=(--resume)
status "$NAME: train to 10000 on 1 GPU (job $SLURM_JOB_ID) ${RESUME[*]:-}"
"$ROOT/external/openpi/.venv/bin/python" "$ROOT/adaspe_train/train.py" "$NAME" --exp_name formal "${RESUME[@]}" \
  --fsdp_devices 1 >> "$RUN/${NAME#${TAG}_}.log" 2>&1
[[ -d "$DIR/9999/params" ]] || { status "$NAME FAILED: no step-9999 params"; exit 1; }
status "$NAME done"
