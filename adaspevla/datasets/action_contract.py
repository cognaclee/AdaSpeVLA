"""Explicit physical action contract; no inferred frames or dimension padding."""

import numpy as np
import torch


def canonicalize_episode(ep, action_dim):
    """Return base-frame twists, binary grippers, and timestamps in seconds.

    Each arm is (vx,vy,vz,wx,wy,wz,gripper), with angular velocity expressed as
    an axis-angle rate, not Euler-angle rates. delta_twist input contains spatial
    translation/rotation increments over explicitly supplied action durations.
    Joint commands, normalized controller actions, poses and tool-frame commands
    must be converted by the caller before entering this contract.
    """
    if action_dim not in (7, 14):
        raise ValueError("Only explicit 7- or 14-channel arm actions are supported")
    actions = np.asarray(ep["actions"], dtype=np.float64)
    if (
        actions.ndim != 2
        or actions.shape[1] != action_dim
        or not np.isfinite(actions).all()
    ):
        raise ValueError(
            "Actions must be finite (T, action_dim); padding/truncation is not supported"
        )
    T = len(actions)
    if T < 2:
        raise ValueError("An episode needs at least two action samples")
    if ep.get("action_frame") != "base" or ep.get("action_type") not in (
        "twist",
        "delta_twist",
    ):
        raise ValueError(
            "Declare action_frame=base and action_type=twist or delta_twist"
        )
    sample_dt = ep.get("sample_dt")
    if sample_dt is not None and (not np.isfinite(sample_dt) or sample_dt <= 0):
        raise ValueError("sample_dt must be finite and positive seconds")
    if "timestamps" in ep:
        times = np.asarray(ep["timestamps"], dtype=np.float64)
    elif sample_dt is not None:
        times = np.arange(T, dtype=np.float64) * sample_dt
    else:
        raise ValueError("Explicit timestamps or sample_dt in seconds are required")
    if (
        times.shape != (T,)
        or not np.isfinite(times).all()
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError("timestamps must have length T and strictly increase")
    if sample_dt is not None and not np.allclose(
        np.diff(times), sample_dt, rtol=1e-4, atol=1e-8
    ):
        raise ValueError("sample_dt contradicts timestamps")
    arms = actions.reshape(T, action_dim // 7, 7)
    motion = arms[..., :6].copy()
    if ep["action_type"] == "delta_twist":
        if "action_durations" in ep:
            durations = np.asarray(ep["action_durations"], dtype=np.float64)
        elif sample_dt is not None:
            durations = np.full(T, sample_dt)
        else:
            raise ValueError(
                "delta_twist requires sample_dt or a duration for every action, including the last"
            )
        if (
            durations.shape != (T,)
            or not np.isfinite(durations).all()
            or (durations <= 0).any()
        ):
            raise ValueError("Invalid action_durations")
        motion /= durations[:, None, None]
    if "gripper_open_value" not in ep or "gripper_closed_value" not in ep:
        raise ValueError("Declare gripper_open_value and gripper_closed_value")
    opened, closed = float(ep["gripper_open_value"]), float(ep["gripper_closed_value"])
    if not np.isfinite([opened, closed]).all() or opened == closed:
        raise ValueError("Gripper endpoints must be finite and distinct")
    raw_gripper = arms[..., 6]
    is_open, is_closed = np.isclose(raw_gripper, opened), np.isclose(
        raw_gripper, closed
    )
    if not (is_open | is_closed).all():
        raise ValueError(
            "Grippers must be discrete open/closed commands; convert position controls explicitly"
        )
    return motion.reshape(T, -1).astype(np.float32), is_open.astype(np.float32), times


def calibrate_motion_scales(motions, floor=1e-6):
    """95th absolute percentile per channel, pooling arms on the training split."""
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("Scale floor must be positive")
    values = np.concatenate([np.asarray(x).reshape(-1, 6) for x in motions])
    return torch.tensor(
        np.maximum(np.quantile(np.abs(values), 0.95, axis=0), floor),
        dtype=torch.float32,
    )


def validate_motion_scales(values, num_arms):
    scales = torch.as_tensor(values, dtype=torch.float32)
    if scales.shape != (6,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError(
            "motion_scale_values must contain six positive finite training scales"
        )
    return scales.repeat(num_arms)
