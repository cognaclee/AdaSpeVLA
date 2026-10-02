"""Nonuniform knot placement and supervised query-interval prediction."""

import torch
from torch import nn
from torch.nn import functional as F


def intervals_to_knots(delta: torch.Tensor, degree: int = 3) -> torch.Tensor:
    if delta.ndim != 2 or degree < 0:
        raise ValueError("Expected batched intervals and a nonnegative degree")
    return torch.cat(
        (
            delta.new_zeros(delta.shape[0], degree + 1),
            delta.cumsum(-1)[:, :-1],
            delta.new_ones(delta.shape[0], degree + 1),
        ),
        dim=-1,
    )


def duration_tensor(value, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a physical duration in seconds to (B, 1)."""
    duration = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    duration = duration.reshape(-1, 1).expand(reference.shape[0], 1)
    if not torch.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("Horizon duration must be finite and positive")
    return duration


class LearnableKnotPlacementHead(nn.Module):
    """Learn spans and query timing; physical horizon is supplied, not predicted."""

    def __init__(
        self,
        cond_dim=512,
        num_control_points=8,
        degree=3,
        hidden_dim=256,
        temperature=1.0,
        min_interval_eps=0.01,
        min_query_interval=0.02,
        max_query_interval=0.5,
        num_groups=1,
    ):
        super().__init__()
        self.num_intervals = num_control_points - degree
        if num_groups < 1:
            raise ValueError("Require at least one knot group")
        # num_groups > 1 gives one span distribution per arm from a shared trunk.
        self.num_groups = num_groups
        if degree < 0 or self.num_intervals < 1 or temperature <= 0:
            raise ValueError("Require K > degree >= 0 and positive temperature")
        if not 0 <= min_interval_eps < 1 / self.num_intervals:
            raise ValueError("Minimum span must lie in [0, 1/(K-degree))")
        if not 0 < min_query_interval < max_query_interval:
            raise ValueError("Require 0 < minimum query interval < maximum")
        self.degree = degree
        self.temperature = temperature
        self.min_interval_eps = min_interval_eps
        self.min_query_interval = min_query_interval
        self.max_query_interval = max_query_interval
        self.net = nn.Sequential(
            nn.Linear(cond_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_intervals * num_groups),
        )

    def forward(self, cond, horizon_duration=0.3):
        horizon = duration_tensor(horizon_duration, cond)
        logits = self.net(torch.cat((cond, horizon.log()), dim=-1))
        if self.num_groups > 1:
            logits = logits.reshape(-1, self.num_groups, self.num_intervals)
        raw = F.softmax(logits / self.temperature, dim=-1)
        # This parameter is an actual lower bound, including after normalization.
        delta = (
            self.min_interval_eps
            + (1 - self.num_intervals * self.min_interval_eps) * raw
        )
        if self.num_groups == 1:
            return intervals_to_knots(delta, self.degree), delta
        knots = intervals_to_knots(delta.flatten(0, 1), self.degree)
        return knots.reshape(delta.shape[0], self.num_groups, -1), delta


def compute_curvature_density_target(
    actions,
    num_intervals=5,
    timestamps=None,
    min_interval_eps=0.01,
    density_gain=10.0,
    criticality=None,
):
    """Inverse-CDF spans from normalized motion velocities (grippers excluded).

    Acceleration magnitude is a representation-complexity heuristic, not measured
    feedback need. Timestamps are physical seconds; optional criticality is [0,1].
    The density is constant on each observed time interval, so short and irregular
    trajectories do not require empty time bins.
    """
    if actions.ndim != 3 or actions.shape[1] < 2 or num_intervals < 1:
        raise ValueError("Need (B,T,D), T >= 2 and at least one knot interval")
    if not 0 <= min_interval_eps < 1 / num_intervals:
        raise ValueError("Invalid minimum knot span")
    B, T, _ = actions.shape
    if timestamps is None:
        raise ValueError(
            "Physical timestamps in seconds are required for density targets"
        )
    timestamps = timestamps.to(actions)
    widths = timestamps.diff(dim=-1)
    if (
        timestamps.shape != (B, T)
        or not torch.isfinite(timestamps).all()
        or (widths <= 0).any()
    ):
        raise ValueError(
            "Timestamps must be finite, strictly increasing and shaped (B,T)"
        )
    if not torch.isfinite(actions).all():
        raise ValueError("Motion values must be finite")
    accel = (actions.diff(dim=1) / widths.unsqueeze(-1)).norm(dim=-1)
    density = 1 + density_gain * torch.tanh(accel)
    if criticality is not None:
        if (
            criticality.shape != (B, T)
            or not torch.isfinite(criticality).all()
            or ((criticality < 0) | (criticality > 1)).any()
        ):
            raise ValueError("criticality must be finite (B,T) values in [0,1]")
        density = density + density_gain * torch.maximum(
            criticality[:, :-1], criticality[:, 1:]
        )
    masses = density * widths
    cdf = torch.cat((actions.new_zeros(B, 1), masses.cumsum(-1)), -1)
    cdf = cdf / cdf[:, -1:]
    quantiles = (
        torch.linspace(
            0, 1, num_intervals + 1, device=actions.device, dtype=actions.dtype
        )
        .expand(B, -1)
        .contiguous()
    )
    index = (
        torch.searchsorted(cdf.contiguous(), quantiles, right=True)
        .sub(1)
        .clamp(0, T - 2)
    )
    c0, c1 = cdf.gather(1, index), cdf.gather(1, index + 1)
    t0, t1 = timestamps.gather(1, index), timestamps.gather(1, index + 1)
    breaks = t0 + (quantiles - c0) / (c1 - c0) * (t1 - t0)
    delta = breaks.diff(dim=-1) / (timestamps[:, -1:] - timestamps[:, :1])
    return min_interval_eps + (1 - num_intervals * min_interval_eps) * delta


def query_interval_target(
    delta,
    horizon,
    min_interval=0.02,
    max_interval=0.5,
    timestamps=None,
    criticality=None,
    gripper=None,
    motion=None,
    e_crit=None,
    anticipation_margin=0.0,
):
    """Event-based timing teacher; knot span lengths do not determine the label.

    Measured open-loop query-interval labels should override this teacher in data.
    """
    if motion is None or timestamps is None:
        raise ValueError(
            "Normalized motion and physical timestamps are required for timing targets"
        )
    if motion.ndim != 3 or motion.shape[1] < 2 or not torch.isfinite(motion).all():
        raise ValueError("Motion must be finite (B,T,D) with T >= 2")
    B, T, _ = motion.shape
    if (
        delta.ndim != 2
        or delta.shape[0] != B
        or delta.shape[1] < 1
        or not torch.isfinite(delta).all()
        or (delta < 0).any()
        or not torch.allclose(delta.sum(-1), torch.ones_like(delta[:, 0]), atol=1e-5)
    ):
        raise ValueError("Target spans must be nonnegative and sum to one")
    timestamps = timestamps.to(delta)
    if (
        timestamps.shape != (B, T)
        or not torch.isfinite(timestamps).all()
        or (timestamps.diff(dim=-1) <= 0).any()
    ):
        raise ValueError(
            "Physical timestamps must be finite and strictly increasing (B,T)"
        )
    if not 0 < min_interval < max_interval < float("inf"):
        raise ValueError("Require finite 0 < minimum interval < maximum interval")
    horizon = duration_tensor(horizon, delta)
    if not torch.allclose(timestamps[:, -1:] - timestamps[:, :1], horizon):
        raise ValueError("Horizon must equal the demonstrated timestamp duration")
    if criticality is not None and (
        criticality.shape != (B, T)
        or not torch.isfinite(criticality).all()
        or ((criticality < 0) | (criticality > 1)).any()
    ):
        raise ValueError("Criticality must be finite (B,T) values in [0,1]")
    if gripper is not None and (
        gripper.ndim != 3
        or gripper.shape[:2] != (B, T)
        or not ((gripper == 0) | (gripper == 1)).all()
    ):
        raise ValueError("Demonstrated gripper states must be binary (B,T,arms)")
    lower = torch.minimum(horizon, torch.full_like(horizon, min_interval))
    upper = torch.minimum(horizon, torch.full_like(horizon, max_interval))
    if e_crit is None or not 0 < e_crit < float("inf"):
        raise ValueError("A positive training-calibrated e_crit is required")
    if not 0 <= anticipation_margin < float("inf"):
        raise ValueError("Anticipation margin must be finite and nonnegative")
    variation_rate = (motion.diff(dim=1) / timestamps.diff(dim=-1).unsqueeze(-1)).norm(
        dim=-1
    )
    # Only j >= 1 is eligible, as specified by the manuscript event set.
    event = variation_rate >= e_crit
    if criticality is not None:
        event |= criticality[:, 1:] >= 0.5
    if gripper is not None:
        event |= (gripper[:, 1:] != gripper[:, :-1]).any(-1)
    relative_times = timestamps[:, 1:] - timestamps[:, :1]
    risk_time = torch.where(
        event, relative_times, torch.full_like(relative_times, float("inf"))
    ).amin(-1, keepdim=True)
    return (risk_time - anticipation_margin).clamp(min=0).clamp(min=lower, max=upper)
