# Sourced by run_train.sh and run_timing_one.sh (needs TASK, JOB_BASE from names.sh, and status()).
# Dev bridge: while the 20 h t3 chain (job name adaspe-<TASK>) waits in the queue, the same train_task.slurm runs as
# 2 h links on qos_gpu_h100-dev (job name adaspe-dev-<TASK>), each resuming from the last save. Any other caller (the
# t3 link) first cancels the dev chain and waits until its running link is gone, so two jobs never write the same
# checkpoint dir. The dev link itself skips this.

_dev_jobs() {  # ids of this task's dev-chain jobs; $1 = squeue state filter (default: all active). Retries a failing squeue.
  local out _
  for _ in 1 2 3 4 5; do
    out=$(squeue -h -u "$USER" -n "${JOB_BASE:-adaspe}-dev-$TASK" ${1:+-t "$1"} -o %A 2>/dev/null) && { echo "$out"; return 0; }
    sleep 20
  done
  return 1
}

stop_dev_chain() {
  [[ "${SLURM_JOB_NAME:-}" == "${JOB_BASE:-adaspe}-dev-$TASK" ]] && return 0
  local ids
  ids=$(_dev_jobs) || { status "squeue keeps failing: dev chain ${JOB_BASE:-adaspe}-dev-$TASK not checked"; return 0; }
  [[ -z "$ids" ]] && return 0
  status "stopping dev chain ${JOB_BASE:-adaspe}-dev-$TASK (jobs $(echo $ids)) before training here"
  scancel $ids 2>/dev/null || true
  # A failing squeue counts as "still running": never start while a dev link may still be writing.
  while ids=$(_dev_jobs RUNNING,COMPLETING,CONFIGURING) || ids=unknown; [[ -n "$ids" ]]; do sleep 15; done
  sleep 30
  ids=$(_dev_jobs) && [[ -n "$ids" ]] && { scancel $ids 2>/dev/null || true; }  # a successor submitted meanwhile
  status "dev chain ${JOB_BASE:-adaspe}-dev-$TASK stopped"
}
