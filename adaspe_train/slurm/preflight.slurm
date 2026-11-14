#!/usr/bin/bash
#SBATCH --job-name=adaspe-preflight
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=01:30:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --hint=nomultithread
#SBATCH --exclude=jzxh008,jzxh043,jzxh045,jzxh018
#SBATCH --output=test/franka/%x-%j.out
# run_train.sh preflight only (train 30 steps, resume to 40, finite losses, resume continues from step 30) on the
# same 4 GPUs / FSDP 4 as the formal Action run. Writes test/franka/<TASK>_adaspe/preflight_passed, which the
# train_task.slurm chain then skips. Submit from the repo root; the job name must be adaspe-pf-<TASK>
# (run_train.sh waits for it):
#   sbatch -A <account> --job-name=adaspe-pf-insert_tube --export=ALL,TASK=insert_tube adaspe_train/slurm/preflight.slurm
set -uo pipefail
unset SLURM_CPU_BIND
: "${TASK:?pass --export=ALL,TASK=<task>}"
ROOT=$SLURM_SUBMIT_DIR  # Slurm runs a spooled copy of this file: the repo root is the submit dir
[[ -f "$ROOT/adaspe_train/tasks.py" ]] || { echo "submit from the repo root (submit dir: $ROOT)"; exit 1; }
export VARIANT=${VARIANT:-}
source "$ROOT/adaspe_train/names.sh"  # TAG, JOB_BASE (VARIANT=expert: job name adaspe-expert-pf-<TASK>)
# Never run past the preflight: with preflight_passed present run_train.sh would go on to the formal Action stage.
[[ -f "$ROOT/test/franka/$TAG/preflight_passed" ]] && { echo "preflight $TAG already passed"; exit 0; }
# The chain's own first link does the preflight; do not race it on the same preflight dir.
[[ -n "$(squeue -h -u "$USER" -n "$JOB_BASE-$TASK" -t RUNNING -o %A)" ]] && { echo "chain $JOB_BASE-$TASK running"; exit 0; }
nvidia-smi -L
CUDA_VISIBLE_DEVICES=0,1,2,3 PREFLIGHT_ONLY=1 TASK=$TASK bash "$ROOT/adaspe_train/run_train.sh"
status=$?
[[ -f "$ROOT/test/franka/$TAG/preflight_passed" ]] || status=1
echo "preflight $TASK exit=$status"
exit $status
