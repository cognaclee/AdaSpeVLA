#!/usr/bin/bash
set -euo pipefail
module purge
# ARCH_MODULE selects the node type's module tree; empty means the default
# (V100) tree. Defaults to H100.
ARCH_MODULE="${ARCH_MODULE-arch/h100}"
module load ${ARCH_MODULE:+$ARCH_MODULE} cuda/12.1.0 anaconda-py3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ADASPEVLA_CONDA_ENV:-grassmann}"
export ADASPEVLA_ROOT="${SLURM_SUBMIT_DIR:?Submit from the AdaSpeVLA repository root}"
cd "$ADASPEVLA_ROOT"
TRAINING_PYTHON="${TRAINING_PYTHON:-$ADASPEVLA_ROOT/tmp/train-env/bin/python}"
[[ -x "$TRAINING_PYTHON" ]] || { echo "Create tmp/train-env or set TRAINING_PYTHON to a compatible training interpreter" >&2; exit 1; }
export PATH="$(dirname "$TRAINING_PYTHON"):$PATH"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/lustre/fswork/projects/rech/sdh/ueh86le/GrassmannPolicy/RoboTwin}"
export PYTHONPATH="$ADASPEVLA_ROOT:$ADASPEVLA_ROOT/robocasa:$ADASPEVLA_ROOT/robosuite:$(dirname "$ROBOTWIN_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
export SAPIEN_NO_DRM=1 EGL_PLATFORM=device MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export USE_TF=0 USE_FLAX=0
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$ADASPEVLA_ROOT/tmp/cache"
# Pretrained backbones (SigLIP) are cached in the home Hugging Face cache.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$XDG_CACHE_HOME"
TASK_MANIFEST="${TASK_MANIFEST:-$ADASPEVLA_ROOT/scripts/benchmarks/slurm_tasks.tsv}"
# TASK_ROW lets array jobs that are not one-task-per-index pick their manifest row.
read -r BENCHMARK TASK_NAME < <(sed -n "$((${TASK_ROW:-$SLURM_ARRAY_TASK_ID} + 1))p" "$TASK_MANIFEST")
[[ "$BENCHMARK" == robotwin || "$BENCHMARK" == robocasa ]] || { echo "Invalid manifest row" >&2; exit 1; }
export BENCHMARK TASK_NAME
RUN_ROOT="${RUN_ROOT:-$ADASPEVLA_ROOT/checkpoints/benchmark-suite}"
CHECKPOINT_DIR="$RUN_ROOT/$BENCHMARK/$TASK_NAME"
CHECKPOINT="$CHECKPOINT_DIR/adaspevla_${BENCHMARK}_${TASK_NAME}_latest.pt"
METRICS_ROOT="${METRICS_ROOT:-$ADASPEVLA_ROOT/eval_metrics/benchmark-suite}"
