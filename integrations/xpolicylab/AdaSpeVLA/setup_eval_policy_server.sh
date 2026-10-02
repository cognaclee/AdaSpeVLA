#!/bin/bash
# Adapted from XPolicyLab policy/demo_policy (Apache-2.0).
set -euo pipefail
bench_name=${1}; task_name=${2}; ckpt_name=${3}; env_cfg_type=${4}; action_type=${5}
seed=${6}; policy_gpu_id=${7}; policy_conda_env=${8}; policy_server_port=${9}
policy_server_host=${10:-"localhost"}
source "$(dirname "${BASH_SOURCE[0]}")/_paths.sh"
policy_name="$(basename "${POLICY_DIR}")"
yaml_file="${POLICY_DIR}/deploy.yml"
echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}"
if [[ -n "${policy_conda_env}" && "${policy_conda_env}" != "none" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
fi
exec env PYTHONWARNINGS=ignore::UserWarning CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" host="${policy_server_host}" \
            bench_name="${bench_name}" task_name="${task_name}" ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" seed="${seed}" policy_name="${policy_name}" \
            action_type="${action_type}"
