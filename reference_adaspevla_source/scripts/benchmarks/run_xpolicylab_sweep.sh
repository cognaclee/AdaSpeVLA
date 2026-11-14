#!/bin/bash
# Adaptive-vs-fixed query timing sweep on RoboTwin 2.0 or RoboDojo via XPolicyLab.
#
# Usage: bash scripts/benchmarks/run_xpolicylab_sweep.sh <XPolicyLab/policy/AdaSpeVLA> \
#            <bench_name> <env_cfg_type> <seed> <policy_gpu> <env_gpu> \
#            <policy_conda_env> <eval_conda_env> <task> [task ...]
# Environment:
#   FIXED_INTERVALS  fixed query intervals in seconds (default: one, four and
#                    eight RoboTwin samples at 0.06 s, plus the 0.5 s timing cap)
#   EXTRA_INFO       benchmark options, e.g. "task_config=demo_clean,test_num=100"
#   METRICS_DIR      output for per-episode JSONL (default: <policy_dir>/eval_metrics)
# Each task's checkpoint is resolved from ckpt_name=<task> (train.sh naming).
set -euo pipefail
policy_dir="$(cd "$1" && pwd)"; bench=$2; env_cfg=$3; seed=$4; pgpu=$5; egpu=$6
penv=$7; eenv=$8; shift 8
intervals=${FIXED_INTERVALS:-"0.06 0.24 0.48 0.5"}
export ADASPEVLA_METRICS_DIR="${METRICS_DIR:-${policy_dir}/eval_metrics}"
for task in "$@"; do
    ADASPEVLA_QUERY_MODE=adaptive bash "${policy_dir}/eval.sh" "${bench}" "${task}" "${task}" \
        "${env_cfg}" ee "${seed}" "${pgpu}" "${egpu}" "${penv}" "${eenv}"
    for interval in ${intervals}; do
        ADASPEVLA_QUERY_MODE=fixed ADASPEVLA_FIXED_QUERY_INTERVAL="${interval}" \
            bash "${policy_dir}/eval.sh" "${bench}" "${task}" "${task}" \
            "${env_cfg}" ee "${seed}" "${pgpu}" "${egpu}" "${penv}" "${eenv}"
    done
done
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m scripts.benchmarks.summarize_metrics "${ADASPEVLA_METRICS_DIR}"
