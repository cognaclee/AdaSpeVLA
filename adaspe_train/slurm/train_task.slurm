#!/usr/bin/bash
#SBATCH --job-name=adaspe-franka
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --hint=nomultithread
#SBATCH --dependency=singleton
#SBATCH --exclude=jzxh008,jzxh043,jzxh045,jzxh018
#SBATCH --output=test/franka/%x-%j.out
# Full AdaSpeVLA pipeline for one task (README recipe), as a chain of 20 h H100 links on one 4-GPU node:
#   preflight (30 steps + resume) -> Action 40k on 4 GPUs (FSDP 4, global batch 32; resumes from the last
#   1000-step save) -> the 3 timing runs (from 20k/30k/40k) in parallel, one GPU each.
# Each link queues its successor (same job name, singleton); the link that finds everything done, or that hits a
# real failure (not the time limit), cancels it. Never change the Action GPU count once it has started.
# Start from the repo root (ROOT and the log path are the submit dir), with your account; later links reuse it:
#   sbatch -A <account> --job-name=adaspe-insert_tube     --export=ALL,TASK=insert_tube     adaspe_train/slurm/train_task.slurm
#   sbatch -A <account> --job-name=adaspe-wipe_whiteboard --export=ALL,TASK=wipe_whiteboard adaspe_train/slurm/train_task.slurm
# Logs: test/franka/<job name>-<id>.out (chain) and test/franka/<TASK>_adaspe/ (pipeline_status.log, action.log, ...).
# Checkpoints: checkpoints/adaspe/<TASK>_adaspe_*/formal (symlink to $SCRATCH).
set -uo pipefail
unset SLURM_CPU_BIND
ROOT=$SLURM_SUBMIT_DIR  # Slurm runs a spooled copy of this file: the repo root is the submit dir
[[ -f "$ROOT/adaspe_train/tasks.py" ]] || { echo "submit from the repo root (submit dir: $ROOT)"; exit 1; }
SELF=$ROOT/adaspe_train/slurm/train_task.slurm
: "${TASK:?pass --export=ALL,TASK=<task>}"
export TASK
CK=$ROOT/checkpoints/adaspe
RUN=$ROOT/test/franka/${TASK}_adaspe
mkdir -p "$RUN"
log() { echo "$(date -Is) [chain $SLURM_JOB_ID] $*" | tee -a "$RUN/pipeline_status.log"; }
action_done() { [[ -d "$CK/${TASK}_adaspe_action/formal/39999/params" ]]; }
timing_done() { [[ -d "$CK/${TASK}_adaspe_timing_from$1k/formal/9999/params" ]]; }
all_done() { action_done && timing_done 20 && timing_done 30 && timing_done 40; }
queued() { squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t PENDING -o %A; }
cancel_successor() { for _ in 1 2 3; do sleep 20; q=$(queued); [[ -n "$q" ]] && scancel $q; done; }

if all_done; then log "all stages already done"; cancel_successor; exit 0; fi
( for _ in $(seq 90); do
      [[ -n "$(queued)" ]] && break
      sbatch -A "$SLURM_JOB_ACCOUNT" --job-name="$SLURM_JOB_NAME" --export=ALL,TASK="$TASK" "$SELF" && break
      sleep 60
  done ) &
RETRY=$!

END=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)
left() { echo $(( END - $(date +%s) - 600 )); }  # stop 10 min before the limit; saves come every 1000 steps
log "link start on $(hostname), GPUs: $(nvidia-smi -L | wc -l), $(( $(left) / 60 )) min usable"
status=0
if ! action_done; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 ACTION_ONLY=1 timeout -k 120 "$(left)" bash "$ROOT/adaspe_train/run_train.sh"
  status=$?
fi
if (( status == 0 )) && action_done; then
  pids=()
  for i in 0 1 2; do
    k=$(( 20 + 10 * i ))
    timing_done $k && continue
    CUDA_VISIBLE_DEVICES=$i timeout -k 120 "$(left)" bash "$ROOT/adaspe_train/run_timing_one.sh" $k &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || { s=$?; (( status == 0 || status == 124 )) && status=$s; }; done
fi

kill "$RETRY" 2>/dev/null
if all_done; then
  log "ALL DONE: action 20k/30k/40k + 3 timing heads"
  cancel_successor
elif (( status == 124 || status == 137 )); then
  log "time limit reached (status $status); the queued link resumes"
else
  log "FAILED (status $status); chain stopped, see $RUN/*.log"
  cancel_successor
fi
exit "$status"
