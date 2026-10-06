"""Anchor-relative pose offsets in an explicit robot base frame.

Poses are ``[x, y, z, qw, qx, qy, qz]`` (XPolicyLab/RoboTwin order) unless a
function says otherwise. The motion action is the offset of a pose from the
anchor pose observed at query time (manuscript Eq. anchor_delta), expressed in
the arm base frame with an axis-angle rotation vector. ``delta_to_target`` is
its exact inverse (Eq. delta_to_target), so converting demonstrated poses to
offsets and back reproduces the demonstrated poses.
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


def anchor_deltas(poses, anchor, world_from_base):
    """Anchor-relative base-frame offsets (Eq. anchor_delta), (T,7) -> (T,6).

    ``poses`` and ``anchor`` are ``[x,y,z,qw,qx,qy,qz]`` in the logging frame;
    ``world_from_base`` (R_b) rotates base-frame vectors into that frame. The
    first three channels are ``R_b^T (p - p_anchor)`` in metres, the last three
    the rotation vector ``R_b^T Log(R R_anchor^T)`` in radians (a spatial,
    left-composed relative rotation, not Euler angles).
    """
    poses = np.asarray(poses, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7 or anchor.shape != (7,):
        raise ValueError("Expected (T,7) poses and a 7-D anchor pose")
    if not np.isfinite(poses).all() or not np.isfinite(anchor).all():
        raise ValueError("Poses must be finite")
    rotations = wxyz_to_rotation(poses[:, 3:], unit_atol=1e-3)
    reference = wxyz_to_rotation(anchor[3:], unit_atol=1e-3)
    to_base = world_from_base.inv()
    position = to_base.apply(poses[:, :3] - anchor[:3])
    rotation = to_base.apply((rotations * reference.inv()).as_rotvec())
    return np.concatenate((position, rotation), axis=-1)


def delta_to_target(anchor, offset, world_from_base):
    """Absolute pose target from an anchor-relative offset (Eq. delta_to_target).

    Exact inverse of :func:`anchor_deltas`: ``p = p_anchor + R_b a_pos`` and
    ``R = Exp(R_b a_rot) R_anchor``. ``offset`` may be (6,) or (N,6).
    """
    anchor = np.asarray(anchor, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    single = offset.ndim == 1
    offset = offset.reshape(-1, 6) if offset.shape[-1:] == (6,) else None
    if anchor.shape != (7,) or offset is None:
        raise ValueError("Expected a 7-D anchor pose and 6-D offsets")
    if not np.isfinite(anchor).all() or not np.isfinite(offset).all():
        raise ValueError("Anchor pose and offsets must be finite")
    reference = wxyz_to_rotation(anchor[3:])
    position = anchor[:3] + world_from_base.apply(offset[:, :3])
    rotation = Rotation.from_rotvec(world_from_base.apply(offset[:, 3:])) * reference
    target = np.concatenate((position, rotation_to_wxyz(rotation)), axis=-1)
    return target[0] if single else target


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
