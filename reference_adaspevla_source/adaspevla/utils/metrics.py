"""Physical trajectory diagnostics and query-count comparisons."""

import math
import numpy as np
import torch


def _array(value):
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


def compute_trajectory_jerk(trajectory, dt=0.02, representation="position"):
    """Integral squared physical jerk; exclude discrete grippers and separate units.

    Position requires three differences; physical velocity requires two. Linear
    and angular channels should be reported separately, not summed as one metric.
    """
    if (
        not math.isfinite(dt)
        or dt <= 0
        or representation not in ("position", "velocity")
    ):
        raise ValueError("Invalid time step or trajectory representation")
    values = _array(trajectory)
    order = 3 if representation == "position" else 2
    if not np.isfinite(values).all():
        raise ValueError("Trajectory must be finite")
    if len(values) <= order:
        return 0.0
    jerk = np.diff(values, n=order, axis=0) / dt**order
    return float(np.sum(jerk**2) * dt)


def compute_chunk_boundary_jump(chunk_prev, chunk_next, dt_prev=0.02, dt_next=0.02):
    """One-sided finite-difference estimates of value and time-derivative jumps."""
    if min(dt_prev, dt_next) <= 0 or not np.isfinite([dt_prev, dt_next]).all():
        raise ValueError("Time steps must be positive and finite")
    prev, nxt = _array(chunk_prev), _array(chunk_next)
    if len(prev) < 3 or len(nxt) < 3:
        raise ValueError("At least three samples per chunk are required")
    return {
        "c0_value_jump": float(np.linalg.norm(nxt[0] - prev[-1])),
        "c1_derivative_jump": float(
            np.linalg.norm(
                (nxt[1] - nxt[0]) / dt_next - (prev[-1] - prev[-2]) / dt_prev
            )
        ),
        "c2_derivative_jump": float(
            np.linalg.norm(
                (nxt[2] - 2 * nxt[1] + nxt[0]) / dt_next**2
                - (prev[-1] - 2 * prev[-2] + prev[-3]) / dt_prev**2
            )
        ),
    }


def compute_efficiency_savings(adaptive_query_hz, baseline_query_hz=20.0):
    """Query-count reduction only; does not estimate GPU cost or speedup."""
    if (
        not np.isfinite([adaptive_query_hz, baseline_query_hz]).all()
        or adaptive_query_hz < 0
        or baseline_query_hz <= 0
    ):
        raise ValueError("Invalid query rates")
    return {
        "adaptive_query_hz": float(adaptive_query_hz),
        "baseline_query_hz": float(baseline_query_hz),
        "query_reduction_pct": float(100 * (1 - adaptive_query_hz / baseline_query_hz)),
    }
