#!/usr/bin/env bash
# Wait until enough GPUs are idle, then launch a command on them.
#
# Usage: wait_gpus_and_launch.sh <num_gpus> <log_file> -- <command...>
# The command sees CUDA_VISIBLE_DEVICES set to the chosen GPUs.
#
# A GPU counts as idle when memory used <= MAX_MEM_MIB and utilization <= MAX_UTIL
# on every check for STABLE_CHECKS checks in a row (other containers' jobs are invisible
# here, so a short dip between their steps must not count).
set -euo pipefail

NUM=${1:?num_gpus}; LOG=${2:?log_file}; shift 2
[[ ${1:-} == -- ]] && shift
[[ $# -gt 0 ]] || { echo "missing command" >&2; exit 2; }

MAX_MEM_MIB=${MAX_MEM_MIB:-2000}
MAX_UTIL=${MAX_UTIL:-5}
INTERVAL=${INTERVAL:-60}
STABLE_CHECKS=${STABLE_CHECKS:-5}
STATUS=${STATUS:-$(dirname "$LOG")/wait_status.txt}
mkdir -p "$(dirname "$LOG")"

declare -A streak
while true; do
  idle=()
  while IFS=', ' read -r idx mem util; do
    if (( mem <= MAX_MEM_MIB && util <= MAX_UTIL )); then
      streak[$idx]=$(( ${streak[$idx]:-0} + 1 ))
    else
      streak[$idx]=0
    fi
    (( streak[$idx] >= STABLE_CHECKS )) && idle+=("$idx")
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

  echo "$(date -Is) waiting: need $NUM idle GPUs, stable idle now: ${idle[*]:-none}" > "$STATUS"
  if (( ${#idle[@]} >= NUM )); then
    gpus=$(IFS=,; echo "${idle[*]:0:NUM}")
    echo "$(date -Is) launching on GPUs $gpus: $*" | tee "$STATUS" >> "$LOG"
    export CUDA_VISIBLE_DEVICES=$gpus
    exec "$@" >> "$LOG" 2>&1
  fi
  sleep "$INTERVAL"
done
