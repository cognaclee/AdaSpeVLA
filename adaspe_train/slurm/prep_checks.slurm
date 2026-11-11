#!/usr/bin/bash
#SBATCH --job-name=adaspe-prep
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:30:00
#SBATCH --hint=nomultithread
#SBATCH --output=test/franka/prep_checks-%j.out
# CPU checks before GPU training (README "运行顺序" 1+3), for insert_tube and wipe_whiteboard:
#   norm stats recomputed and compared with the baseline repo; B-spline parity; model smoke test;
#   data check; timing-label check (0 mismatches). Compute nodes are offline: everything is local.
# Start from the repo root: sbatch -A <account> adaspe_train/slurm/prep_checks.slurm   (SKIP_NORM=1: skip the norm-stats recompute)
set -uo pipefail
unset SLURM_CPU_BIND PYTHONPATH LD_LIBRARY_PATH
ROOT=$SLURM_SUBMIT_DIR  # Slurm runs a spooled copy of this file: the repo root is the submit dir
[[ -f "$ROOT/adaspe_train/tasks.py" ]] || { echo "submit from the repo root (submit dir: $ROOT)"; exit 1; }
A=$ROOT/adaspe_train
PY=$ROOT/external/openpi/.venv/bin/python
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 HF_LEROBOT_HOME=$ROOT/data/lerobot
export HF_HOME=$ROOT/.cache/huggingface OPENPI_DATA_HOME=$ROOT/.cache/openpi TMPDIR=$ROOT/.cache/tmp WANDB_MODE=disabled
mkdir -p "$TMPDIR"
cd "$ROOT/external/openpi"
fail=0
run() { local name=$1; shift; echo "===== $name"; "$@" 2>&1 | grep -v -i -E "warn|it/s\]|s/it\]"; local s=${PIPESTATUS[0]}; echo "===== $name exit=$s"; (( s == 0 )) || fail=1; }
run parity $PY $A/tests/test_bspline_parity.py
run smoke $PY $A/tests/smoke_model.py
for T in ${TASKS:-insert_tube wipe_whiteboard}; do
  export TASK=$T
  REPO=$(TASK=$T $PY -c "import sys; sys.path.insert(0, '$A'); import tasks; print(tasks.SPEC['repo_id'])")
  if [[ "${SKIP_NORM:-0}" != 1 ]]; then  # the norm-stats reference config does not depend on the recipe knobs
    run "$T norm_stats" $PY $A/compute_norm_stats.py --config-name pi05_${T}_franka_ref
    run "$T compare_norm_stats" $PY $A/tests/compare_norm_stats.py \
        $A/assets/pi05_${T}_franka_ref/$REPO/norm_stats.json $A/assets/baseline/$REPO/norm_stats.json
  fi
  run "$T check_data" $PY $A/tests/check_data.py
  run "$T check_timing_labels" $PY $A/tests/check_timing_labels.py
done
echo "ALL_CHECKS fail=$fail"
exit $fail
