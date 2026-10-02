#!/bin/bash
# Install XPolicyLab and AdaSpeVLA (editable) into the active policy environment.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_paths.sh"
python -m pip install -e "${XPL_ROOT}"
python -m pip install -e "${ADASPEVLA_ROOT}"
