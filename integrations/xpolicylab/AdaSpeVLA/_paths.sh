#!/bin/bash
# Shared path resolution. The installer writes .adaspevla_root next to this file.
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
if [[ -z "${ADASPEVLA_ROOT:-}" && -f "${POLICY_DIR}/.adaspevla_root" ]]; then
    ADASPEVLA_ROOT="$(cat "${POLICY_DIR}/.adaspevla_root")"
fi
if [[ -z "${ADASPEVLA_ROOT:-}" || ! -d "${ADASPEVLA_ROOT}/adaspevla" ]]; then
    echo "[AdaSpeVLA] Set ADASPEVLA_ROOT to the AdaSpeVLA repository root" >&2
    exit 1
fi
export ADASPEVLA_ROOT
