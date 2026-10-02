"""Pose differencing and twist integration in an explicit robot base frame.

Poses are ``[x, y, z, qw, qx, qy, qz]`` (XPolicyLab/RoboTwin order) unless a
function says otherwise. Angular velocity is a spatial axis-angle rate, matching
``robotwin_native``: ``omega = Log(R1 @ R0.T) / dt`` expressed in the base frame.
Integration is the exact inverse of that differencing, so converting demonstrated
poses to twists and integrating them back reproduces the demonstrated poses.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def wxyz_to_rotation(quat_wxyz, unit_atol=None):
    """Rotation from wxyz quaternions; optionally reject non-unit inputs."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1)
    if not np.isfinite(q).all() or (norms < 1e-8).any():
        raise ValueError("Quaternions must be finite and nonzero")
    if unit_atol is not None and not np.allclose(norms, 1.0, atol=unit_atol):
        raise ValueError("Pose quaternions must be unit wxyz")
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def rotation_to_wxyz(rotation):
    q = rotation.as_quat()
    return q[..., [3, 0, 1, 2]]


def base_rotation(base_wxyz):
    """World-from-base rotation for a finite nonzero wxyz quaternion."""
    base = np.asarray(base_wxyz, dtype=np.float64)
    if base.shape != (4,):
        raise ValueError("Base orientation must be one wxyz quaternion")
    return wxyz_to_rotation(base)


def pose_pairs_to_twists(current, target, sample_dt, world_from_base):
    """Average base-frame twists between paired poses (T,7) -> (T,6).

    ``current`` and ``target`` are expressed in the same frame (world for
    RoboTwin). ``world_from_base`` rotates base-frame vectors into that frame;
    pass the identity when poses are already in the base frame.
    """
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if current.ndim != 2 or current.shape[1] != 7 or target.shape != current.shape:
        raise ValueError("Expected matching (T,7) current/target poses")
    dt = np.asarray(sample_dt, dtype=np.float64)
    if not np.isfinite(dt).all() or (dt <= 0).any():
        raise ValueError("sample_dt must be positive finite seconds")
    dt = np.broadcast_to(dt, (len(current),))[:, None]
    r0 = wxyz_to_rotation(current[:, 3:], unit_atol=1e-3)
    r1 = wxyz_to_rotation(target[:, 3:], unit_atol=1e-3)
    to_base = world_from_base.inv()
    velocity = to_base.apply((target[:, :3] - current[:, :3]) / dt)
    angular = to_base.apply((r1 * r0.inv()).as_rotvec() / dt)
    return np.concatenate((velocity, angular), axis=-1)


def integrate_twist(pose, twist, dt, world_from_base):
    """Next pose after applying one base-frame twist for ``dt`` seconds."""
    pose = np.asarray(pose, dtype=np.float64)
    twist = np.asarray(twist, dtype=np.float64)
    if pose.shape != (7,) or twist.shape != (6,):
        raise ValueError("Expected a 7-D pose and a 6-D twist")
    if not np.isfinite(twist).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Twist and dt must be finite, dt positive")
    rotation = wxyz_to_rotation(pose[3:])
    position = pose[:3] + world_from_base.apply(twist[:3]) * dt
    delta = Rotation.from_rotvec(world_from_base.apply(twist[3:]) * dt)
    return np.concatenate((position, rotation_to_wxyz(delta * rotation)))


def pose_error(target, actual):
    """Translation error in meters and rotation error in radians."""
    target = np.asarray(target, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    translation = float(np.linalg.norm(target[:3] - actual[:3]))
    rotation = float(
        (wxyz_to_rotation(target[3:]) * wxyz_to_rotation(actual[3:]).inv()).magnitude()
    )
    return translation, rotation


def normalized_pose(pose, unit_atol=0.05):
    """Copy of a pose with its quaternion normalized; reject badly scaled input."""
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Observed end-effector pose must be a finite 7-vector")
    norm = np.linalg.norm(pose[3:])
    if abs(norm - 1.0) > unit_atol:
        raise ValueError(
            f"Observed pose quaternion norm {norm:.4f} is not unit; expected wxyz order"
        )
    return np.concatenate((pose[:3], pose[3:] / norm))
