#!/usr/bin/bash
#SBATCH --job-name=adaspe-tfeat
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
# Offline timing-head features (adaspe_train/dump_timing_features.py). Submit from the repo root:
#   sbatch -A yfl@h100 --job-name=adaspe-tfeat-insert_tube-pi05_expert \
#     --export=ALL,TASK=insert_tube,GENERATOR=pi05_expert adaspe_train/slurm/dump_timing_features.slurm
set -euo pipefail
unset SLURM_CPU_BIND
: "${TASK:?}" "${GENERATOR:?}"
ROOT=$SLURM_SUBMIT_DIR
cd "$ROOT/external/openpi"
export TMPDIR=$ROOT/.cache/tmp HF_HUB_OFFLINE=1 HF_LEROBOT_HOME=$ROOT/data/lerobot
export HF_HOME=$ROOT/.cache/huggingface XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export OPENPI_DATA_HOME=$ROOT/.cache/openpi JAX_COMPILATION_CACHE_DIR=$ROOT/.cache/jax
unset PYTHONPATH LD_LIBRARY_PATH; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
nvidia-smi -L
"$ROOT/external/openpi/.venv/bin/python" "$ROOT/adaspe_train/dump_timing_features.py" \
  --generator "$GENERATOR" --passes "${PASSES:-4}" --workers 20 ${LIMIT:+--limit $LIMIT}
