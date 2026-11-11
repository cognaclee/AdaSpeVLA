"""Geometry for the supplied Franka local rotvec-delta controller.

Pure math only: no robot connection. Collector clipping, delays and compliance
are outside these conversions. Poses use xyz+wxyz in the episode reset frame.
"""
import numpy as np
from scipy.spatial.transform import Rotation

TRANSLATION_SCALE_M = .023
ROTATION_SCALE_RAD = .095


def _pose(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('Expected finite xyz+wxyz pose')
    if not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-3):
        raise ValueError('Unit quaternion required')
    return pose[:3], Rotation.from_quat(pose[[4,5,6,3]])


def body_delta_to_reset_target(current_pose, action):
    p, r = _pose(current_pose)
    action = np.asarray(action, dtype=float)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError('Expected finite seven-channel controller action')
    target_p = p + r.apply(action[:3] * TRANSLATION_SCALE_M)
    target_r = r * Rotation.from_rotvec(action[3:6] * ROTATION_SCALE_RAD)
    return np.r_[target_p, target_r.as_quat()[[3,0,1,2]]]


def reset_target_to_body_delta(current_pose, target_pose):
    p, r = _pose(current_pose)
    target_p, target_r = _pose(target_pose)
    return np.r_[r.inv().apply(target_p - p) / TRANSLATION_SCALE_M,
                 (r.inv() * target_r).as_rotvec() / ROTATION_SCALE_RAD]


def gripper_event(command):
    value = float(command)
    if not np.isfinite(value):
        raise ValueError('Nonfinite gripper command')
    scaled = 1.5 * value
    return 'close' if scaled <= -.5 else 'open' if scaled >= .5 else 'hold'
