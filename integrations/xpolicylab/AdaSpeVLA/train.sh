#!/bin/bash
# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# Trains both AdaSpeVLA stages on process_data.sh output.
# Checkpoints: checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/
# (seed only names the run folder; training itself is not seeded).
# Environment: TASK_NAME (default ckpt_name), TRAIN_ARGS (extra training flags,
#   e.g. "--epochs 20 --timing_epochs 10 --batch_size 32"),
#   OUTPUT_TAG (suffix for the checkpoint folder, e.g. "-per_arm_knots" together
#   with TRAIN_ARGS="--per_arm_knots" for the knot ablation).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_paths.sh"
bench_name=$1; ckpt_name=$2; env_cfg_type=$3; action_type=$4; seed=$5; gpu_id=$6
task_name="${TASK_NAME:-${ckpt_name}}"
setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
cd "${ADASPEVLA_ROOT}"
python -m scripts.benchmarks.train_converted \
    --data_dir "${POLICY_DIR}/data/${setting}" \
    --task_name "${task_name}" \
    --output_dir "${POLICY_DIR}/checkpoints/${setting}-${seed}${OUTPUT_TAG:-}" \
    ${TRAIN_ARGS:-}
