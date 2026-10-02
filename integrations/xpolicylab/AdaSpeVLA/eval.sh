#!/bin/bash
# Usage: bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> ee <seed> \
#            <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
# Environment (forwarded to the policy server):
#   ADASPEVLA_QUERY_MODE=adaptive|fixed, ADASPEVLA_FIXED_QUERY_INTERVAL=<seconds>,
#   ADASPEVLA_METRICS_DIR, ADASPEVLA_CHECKPOINT_PATH, ADASPEVLA_NUM_ODE_STEPS
#   EXTRA_INFO: extra benchmark key=value pairs, e.g.
#     "task_config=demo_randomized,test_num=100" for RoboTwin.
# Adapted from XPolicyLab policy/demo_policy (Apache-2.0).
set -euo pipefail
bench_name=${1}; task_name=${2}; ckpt_name=${3}; env_cfg_type=${4}; action_type=${5}
seed=${6}; policy_gpu_id=${7}; env_gpu_id=${8}; policy_conda_env=${9}; eval_env_conda_env=${10}
source "$(dirname "${BASH_SOURCE[0]}")/_paths.sh"
policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"
if [[ -n "${EXTRA_INFO:-}" ]]; then additional_info="${additional_info},${EXTRA_INFO}"; fi
cleanup() { [[ -n "${SERVER_PID:-}" ]] && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT
bash "${POLICY_DIR}/setup_eval_policy_server.sh" "${bench_name}" "${task_name}" "${ckpt_name}" \
    "${env_cfg_type}" "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_conda_env}" \
    "${policy_server_port}" &
SERVER_PID=$!
bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" \
    "${SERVER_PID}" "Policy server" 1200
bash "${POLICY_DIR}/setup_eval_env_client.sh" "${bench_name}" "${task_name}" "${ckpt_name}" \
    "${env_cfg_type}" "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
    "${additional_info}" "${policy_server_port}" "${policy_server_ip}"
echo "[MAIN] eval finished"
