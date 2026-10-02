#!/bin/bash
# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>
# Converts XPolicyLab trajectories into AdaSpeVLA's physical-twist contract.
# Output: data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/<task_name>/
# Environment:
#   TASK_NAME     task to convert (default: ckpt_name; AdaSpeVLA trains per task)
#   TASK_CONFIG   RoboTwin collection config (default: demo_clean)
#   SOURCE_DIR    episode directory (default: RoboTwin layout
#                 <bench_root>/data/<TASK_CONFIG>/<TASK_NAME>/<env_cfg_type>/data)
#   ARMS          action arms in channel order (default: "left right" for two-arm
#                 robots; use "right" to reproduce the right-arm-only setting)
#   EXTRA_ARGS    extra flags for scripts.benchmarks.convert_xpolicylab
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_paths.sh"
bench_name=$1; ckpt_name=$2; env_cfg_type=$3; action_type=$4
if [[ "${action_type}" != "ee" ]]; then
    echo "[AdaSpeVLA] action_type must be ee" >&2; exit 1
fi
task_name="${TASK_NAME:-${ckpt_name}}"
setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
case "${bench_name}" in
    RoboTwin) profile=robotwin ;;
    RoboDojo) profile=robodojo ;;
    *) echo "[AdaSpeVLA] unsupported bench ${bench_name}" >&2; exit 1 ;;
esac
source_dir="${SOURCE_DIR:-${BENCH_ROOT}/data/${TASK_CONFIG:-demo_clean}/${task_name}/${env_cfg_type}/data}"
cd "${ADASPEVLA_ROOT}"
python -m scripts.benchmarks.convert_xpolicylab \
    --source_dir "${source_dir}" \
    --output_dir "${POLICY_DIR}/data/${setting}" \
    --task_name "${task_name}" \
    --profile "${profile}" \
    --arms ${ARMS:-left right} \
    ${EXTRA_ARGS:-}
