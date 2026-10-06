"""Explicit physical action contract; no inferred frames or dimension padding.

Episodes provide timestamped end-effector poses and gripper widths. Training
targets are anchor-relative offsets (manuscript Eq. anchor_delta) computed per
window by the dataset, plus a continuous normalized gripper width per arm.
"""

import numpy as np
import torch

from ..benchmarks.pose_math import anchor_deltas, base_rotation, delta_to_target

IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)


def _timestamps(ep, T):
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
    return times


def normalize_gripper_width(raw, open_value, closed_value, atol=1e-3):
    """Map raw widths affinely to [0,1] (0 closed, 1 open); never threshold."""
    if open_value is None or closed_value is None:
        raise ValueError("Declare gripper_open_value and gripper_closed_value")
    opened, closed = float(open_value), float(closed_value)
    if not np.isfinite([opened, closed]).all() or opened == closed:
        raise ValueError("Gripper endpoints must be finite and distinct")
    raw = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Gripper widths must be finite")
    width = (raw - closed) / (opened - closed)
    if (width < -atol).any() or (width > 1 + atol).any():
        raise ValueError(
            "Gripper widths lie outside the declared closed/open range; declare the raw range explicitly"
        )
    return np.clip(width, 0.0, 1.0)


def _integrate_motion(motion, increments):
    """Integrate base-frame twists or increments into base-frame poses from identity."""
    T, arms, _ = motion.shape
    poses = np.empty((T, arms, 7))
    identity = base_rotation(IDENTITY_WXYZ)
    for arm in range(arms):
        poses[0, arm] = (0.0, 0.0, 0.0) + IDENTITY_WXYZ
        for j in range(T - 1):
            poses[j + 1, arm] = delta_to_target(
                poses[j, arm], motion[j, arm] * increments[j], identity
            )
    return poses


def canonicalize_episode(ep, action_dim):
    """Return base-frame poses, normalized gripper widths, timestamps and R_b.

    ``action_type='ee_pose'`` supplies ``ee_poses`` (T,arms,7) as
    ``[x,y,z,qw,qx,qy,qz]`` and raw ``gripper`` widths (T,arms); optional
    ``base_wxyz`` (one quaternion or one per arm) is the rotation R_b from the
    arm base frame to the logging frame (identity when poses are base-relative).

    Datasets that log twists or per-step increments are first integrated into
    poses (manuscript Sec. data contract): ``actions`` (T, 7*arms) holds per arm
    six base-frame motion channels (twist: m/s and axis-angle rad/s acting on
    [t_j, t_{j+1}); delta_twist: per-step increments) followed by the raw width.
    The pose at t_0 is the identity; the final sample's motion has no later
    timestamp and is not integrated. Joint commands and normalized controller
    actions must be converted by the caller.

    Returns ``(poses (T,arms,7) float64, widths (T,arms) float32 in [0,1],
    times (T,) seconds, world_from_base list of per-arm rotations)``.
    """
    if action_dim not in (7, 14):
        raise ValueError("Only explicit 7- or 14-channel arm actions are supported")
    arms = action_dim // 7
    if ep.get("action_frame") != "base":
        raise ValueError("Declare action_frame=base (offsets are expressed in the arm base frame)")
    action_type = ep.get("action_type")
    if action_type == "ee_pose":
        if "ee_poses" not in ep or "gripper" not in ep:
            raise ValueError("ee_pose episodes require ee_poses and gripper arrays")
        poses = np.asarray(ep["ee_poses"], dtype=np.float64)
        if poses.ndim == 2 and arms == 1:
            poses = poses[:, None]
        T = len(poses)
        if poses.shape != (T, arms, 7) or not np.isfinite(poses).all():
            raise ValueError("ee_poses must be finite (T, arms, 7) xyz+wxyz poses")
        if T < 2:
            raise ValueError("An episode needs at least two pose samples")
        raw_gripper = np.asarray(ep["gripper"], dtype=np.float64).reshape(T, -1)
        if raw_gripper.shape != (T, arms):
            raise ValueError("gripper must hold one width per arm and sample")
        norms = np.linalg.norm(poses[..., 3:], axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Pose quaternions must be unit wxyz")
        poses = poses.copy()
        poses[..., 3:] /= norms[..., None]
        base = np.asarray(ep.get("base_wxyz", IDENTITY_WXYZ), dtype=np.float64)
        if base.shape == (4,):
            base = np.repeat(base[None], arms, axis=0)
        if base.shape != (arms, 4):
            raise ValueError("base_wxyz must be one quaternion or one per arm")
        world_from_base = [base_rotation(q) for q in base]
        times = _timestamps(ep, T)
    elif action_type in ("twist", "delta_twist"):
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
        times = _timestamps(ep, T)
        per_arm = actions.reshape(T, arms, 7)
        if action_type == "twist":
            increments = np.append(np.diff(times), 0.0)
        else:
            increments = np.ones(T)
        poses = _integrate_motion(per_arm[..., :6], increments)
        raw_gripper = per_arm[..., 6]
        world_from_base = [base_rotation(IDENTITY_WXYZ)] * arms
    else:
        raise ValueError("Declare action_type=ee_pose (or twist/delta_twist, integrated into poses)")
    widths = normalize_gripper_width(
        raw_gripper, ep.get("gripper_open_value"), ep.get("gripper_closed_value")
    )
    return poses, widths.astype(np.float32), times, world_from_base


def window_offsets(poses, world_from_base):
    """Anchor-relative offsets of a pose window (L,arms,7) -> (L, 6*arms)."""
    return np.concatenate(
        [
            anchor_deltas(poses[:, arm], poses[0, arm], rotation)
            for arm, rotation in enumerate(world_from_base)
        ],
        axis=-1,
    )


def calibrate_motion_scales(motions, floor=1e-6):
    """95th absolute percentile per channel over all window samples, pooling arms."""
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


def normalize_actions(actions, motion_scale):
    """Physical (..., 7*arms) offsets+widths -> normalized motion/D and 2w-1."""
    arms = actions.shape[-1] // 7
    blocks = actions.unflatten(-1, (arms, 7))
    scale = motion_scale.to(actions).reshape(arms, 6)
    return torch.cat((blocks[..., :6] / scale, 2 * blocks[..., 6:] - 1), -1).flatten(-2)


def denormalize_actions(values, motion_scale):
    """Normalized (..., 7*arms) -> physical offsets (D x) and widths (x+1)/2, unclamped."""
    arms = values.shape[-1] // 7
    blocks = values.unflatten(-1, (arms, 7))
    scale = motion_scale.to(values).reshape(arms, 6)
    return torch.cat((blocks[..., :6] * scale, (blocks[..., 6:] + 1) / 2), -1).flatten(-2)


def motion_channel_mask(action_dim, device=None):
    """Boolean mask of the six motion channels in each seven-channel arm block."""
    mask = torch.ones(action_dim, dtype=torch.bool, device=device)
    mask[6::7] = False
    return mask
