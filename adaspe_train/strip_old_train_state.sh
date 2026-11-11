#!/bin/bash
# Frees disk: strips train_state (optimizer, resume-only) from kept 10k/20k/30k Action checkpoints once a
# newer step exists, and from finished timing checkpoints (final step 9999). params (used for eval and
# timing) are kept; the latest Action checkpoint stays complete for resume.
# Usage: TASK=bread adaspe_train/strip_old_train_state.sh  (checks every 10 min)
TASK=${TASK:-bread}
CK=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/checkpoints/adaspe
while true; do
  run=$CK/${TASK}_adaspe_action/formal
  latest=$(ls "$run" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  for s in 10000 20000 30000; do
    if [ -n "$latest" ] && [ "$latest" -gt "$s" ] && [ -d "$run/$s/train_state" ] && [ -d "$run/$s/params" ]; then
      rm -rf "$run/$s/train_state" && echo "$(date -Is) stripped $run/$s/train_state"
    fi
  done
  for d in $CK/${TASK}_adaspe_timing_from*/formal/9999; do
    if [ -d "$d/train_state" ] && [ -d "$d/params" ]; then
      rm -rf "$d/train_state" && echo "$(date -Is) stripped $d/train_state"
    fi
  done
  sleep 600
done
