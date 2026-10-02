#!/bin/bash
# Usage: bash scripts/benchmarks/install_xpolicylab_adapter.sh <XPolicyLab checkout>
# Copies integrations/xpolicylab/AdaSpeVLA into <XPolicyLab>/policy/AdaSpeVLA
# (e.g. RoboTwin/XPolicyLab or RoboDojo/XPolicyLab) and records this repository.
# Existing checkpoints/, data/ and eval_metrics/ inside the target are kept.
set -euo pipefail
xpl_root="$(cd "${1:?XPolicyLab checkout path required}" && pwd)"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target="${xpl_root}/policy/AdaSpeVLA"
if [[ ! -f "${xpl_root}/setup_policy_server.py" ]]; then
    echo "${xpl_root} does not look like an XPolicyLab checkout" >&2; exit 1
fi
mkdir -p "${target}"
cp "${repo_root}"/integrations/xpolicylab/AdaSpeVLA/* "${target}/"
cp "${repo_root}"/integrations/xpolicylab/AdaSpeVLA/.[!.]* "${target}/" 2>/dev/null || true
echo "${repo_root}" > "${target}/.adaspevla_root"
chmod +x "${target}"/*.sh
echo "Installed AdaSpeVLA adapter into ${target}"
