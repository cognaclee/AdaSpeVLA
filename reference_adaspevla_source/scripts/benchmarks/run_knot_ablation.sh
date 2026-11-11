#!/bin/bash
# Shared-vs-per-arm knot ablation on bimanual RoboTwin 2.0 (or RoboDojo) via XPolicyLab.
#
# Usage: bash scripts/benchmarks/run_knot_ablation.sh <XPolicyLab/policy/AdaSpeVLA> \
#            <bench_name> <env_cfg_type> <seed> <policy_gpu> <env_gpu> \
#            <policy_conda_env> <eval_conda_env> <task> [task ...]
# Prerequisite: process_data.sh <bench_name> <task> <env_cfg_type> ee with both arms
# (ARMS="left right", the default).
# For each task this trains two checkpoints with identical settings, the
# manuscript's shared knot partition and the --per_arm_knots ablation, then
#   1. compares them offline on the converted windows (eval_knots_offline), and
#   2. evaluates both in simulation with adaptive timing (plus FIXED_INTERVALS).
# Environment:
#   TRAIN_ARGS       flags for both variants, e.g. "--epochs 20 --timing_epochs 10"
#   SKIP_TRAIN=1     reuse existing checkpoints
#   FIXED_INTERVALS  optional fixed query intervals in seconds (default: none)
#   EXTRA_INFO       benchmark options, e.g. "task_config=demo_clean,test_num=100"
#   METRICS_DIR      output directory (default: <policy_dir>/knot_ablation)
set -euo pipefail
policy_dir="$(cd "$1" && pwd)"; bench=$2; env_cfg=$3; seed=$4; pgpu=$5; egpu=$6
penv=$7; eenv=$8; shift 8
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
metrics="${METRICS_DIR:-${policy_dir}/knot_ablation}"
mkdir -p "${metrics}"
export ADASPEVLA_METRICS_DIR="${metrics}"

checkpoint() {  # <task> <tag>
    local dir="${policy_dir}/checkpoints/${bench}-$1-${env_cfg}-ee-${seed}$2"
    local found=( "${dir}"/adaspevla_*_"$1"_latest.pt )
    if [[ ! -f "${found[0]}" ]]; then echo "No checkpoint in ${dir}" >&2; exit 1; fi
    echo "${found[0]}"
}

for task in "$@"; do
    if [[ "${SKIP_TRAIN:-0}" != 1 ]]; then
        TRAIN_ARGS="${TRAIN_ARGS:-}" OUTPUT_TAG="" \
            bash "${policy_dir}/train.sh" "${bench}" "${task}" "${env_cfg}" ee "${seed}" "${pgpu}"
        TRAIN_ARGS="${TRAIN_ARGS:-} --per_arm_knots" OUTPUT_TAG="-per_arm_knots" \
            bash "${policy_dir}/train.sh" "${bench}" "${task}" "${env_cfg}" ee "${seed}" "${pgpu}"
    fi
    shared="$(checkpoint "${task}" "")"
    per_arm="$(checkpoint "${task}" "-per_arm_knots")"
    (cd "${root}" && CUDA_VISIBLE_DEVICES="${pgpu}" python -m scripts.benchmarks.eval_knots_offline \
        --data_dir "${policy_dir}/data/${bench}-${task}-${env_cfg}-ee" --task_name "${task}" \
        --ckpt_paths "${shared}" "${per_arm}" --output_json "${metrics}/offline-${task}.json")
    for ckpt in "${shared}" "${per_arm}"; do
        ADASPEVLA_CHECKPOINT_PATH="${ckpt}" ADASPEVLA_QUERY_MODE=adaptive \
            bash "${policy_dir}/eval.sh" "${bench}" "${task}" "${task}" "${env_cfg}" ee \
            "${seed}" "${pgpu}" "${egpu}" "${penv}" "${eenv}"
        for interval in ${FIXED_INTERVALS:-}; do
            ADASPEVLA_CHECKPOINT_PATH="${ckpt}" ADASPEVLA_QUERY_MODE=fixed \
                ADASPEVLA_FIXED_QUERY_INTERVAL="${interval}" \
                bash "${policy_dir}/eval.sh" "${bench}" "${task}" "${task}" "${env_cfg}" ee \
                "${seed}" "${pgpu}" "${egpu}" "${penv}" "${eenv}"
        done
    done
done
cd "${root}"
python -m scripts.benchmarks.summarize_metrics "${metrics}"
