# Sourced by run_train.sh, run_timing_one.sh and slurm/{train_task,preflight}.slurm after TASK is set.
# VARIANT picks the model variant; every config, checkpoint/log dir and Slurm job name derives from it:
#   VARIANT=        AdaSpeVLA flow head (configs.py generator "flow_head"):
#                   configs <TASK>_adaspe_{action,timing_from{20,30,40}k}, logs test/franka/<TASK>_adaspe,
#                   jobs adaspe-<TASK> (t3 chain), adaspe-dev-<TASK> (dev chain), adaspe-pf-<TASK> (dev preflight)
#   VARIANT=expert  pi0.5 action expert predicts the control points (generator "pi05_expert"):
#                   configs <TASK>_adaspe_expert_*, logs test/franka/<TASK>_adaspe_expert, jobs adaspe-expert-[dev-|pf-]<TASK>
case "${VARIANT:-}" in "" | expert) ;; *) echo "unknown VARIANT=${VARIANT} (use '' or 'expert')" >&2; exit 1 ;; esac
TAG=${TASK}_adaspe${VARIANT:+_$VARIANT}  # config / checkpoint / log-dir prefix
JOB_BASE=adaspe${VARIANT:+-$VARIANT}      # Slurm job-name prefix
