"""Right-arm adapter for RoboTwin's native XPolicyLab v1.0 export (pose contract).

Conventions verified against RoboTwin ea8b21121ebb3cd201ff5b3fe361944ac94eda3f:
  envs/utils/pkl2hdf5.py, envs/robot/robot.py, envs/_base_task.py.
See docs/robotwin_native.md for source links and timing limitations.
"""

import json
import warnings
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_BASE_WXYZ = [0.707, 0.0, 0.0, 0.707]


def xpolicylab_arm_trajectory(group, prefix, label="right"):
    """Observed poses and commanded widths of one arm at the row timestamps.

    ``state/<arm>_ee_poses`` row j is the pose at t_j and ``action/<arm>_ee_poses``
    row j its successor (verified one-frame shift). The width at t_j is the width
    commanded at row j-1 (row 0 for t_0), so a controller step that targets the
    pose at t_{j+1} also receives the width demonstrated for that step. Widths
    stay continuous in [0,1] (1 open); they are never thresholded here.
    """

    def read(path):
        if path not in group:
            raise ValueError(f"XPolicyLab export requires {path}")
        return np.asarray(group[path])

    current = read(f"state/{prefix}ee_poses").astype(np.float64)
    target = read(f"action/{prefix}ee_poses").astype(np.float64)
    if (
        current.ndim != 2
        or current.shape[1] != 7
        or target.shape != current.shape
        or len(current) < 2
    ):
        raise ValueError(f"Expected matching (T,7) {label} state/action poses")
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError(f"{label} poses must be finite")

    def rotations(pose):
        q = pose[:, 3:]
        if not np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-3):
            raise ValueError("Native pose quaternions must be unit wxyz")
        return Rotation.from_quat(q[:, [1, 2, 3, 0]])

    r0, r1 = rotations(current), rotations(target)
    if (
        not np.allclose(target[:-1, :3], current[1:, :3], atol=1e-5)
        or np.max((r1[:-1] * r0[1:].inv()).magnitude()) > 1e-4
    ):
        raise ValueError(
            f"{label} state/action poses do not have the official one-frame shift; refusing to infer alignment"
        )
    grip = read(f"action/{prefix}ee_joint_states")
    if (
        grip.shape != (len(current), 1)
        or not np.isfinite(grip).all()
        or (grip < -1e-6).any()
        or (grip > 1 + 1e-6).any()
    ):
        raise ValueError(
            f"{label} gripper must have shape (T,1) and values in [0,1] (1 open)"
        )
    width = np.concatenate((grip[:1, 0], grip[:-1, 0]))
    poses = current.copy()
    poses[:, 3:] /= np.linalg.norm(poses[:, 3:], axis=-1, keepdims=True)
    return poses, np.clip(width, 0.0, 1.0)


def native_right_episode(
    group,
    images,
    sample_dt=None,
    base_wxyz=None,
    simulation_dt=1 / 250,
):
    """Right-arm observed poses and continuous commanded widths (pose contract).

    The dataset re-anchors these poses at each window start (Eq. anchor_delta).
    """
    poses, width = xpolicylab_arm_trajectory(group, "right_")
    if len(images) != len(poses):
        raise ValueError("Image and right-arm pose lengths differ")
    if "additional_info/frequency" not in group:
        raise ValueError("Native RoboTwin export requires additional_info/frequency")
    frequency = np.asarray(group["additional_info/frequency"])
    if frequency.ndim != 0 or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("additional_info/frequency must be a positive scalar")
    if sample_dt is None:
        # Official native collector exports save_freq, a step stride, not Hz.
        if not np.isfinite(simulation_dt) or simulation_dt <= 0:
            raise ValueError("simulation_dt must be positive")
        sample_dt = float(frequency) * simulation_dt
        time_source = "nominal_collector_stride"
        warnings.warn(
            "Native RoboTwin frequency stores save_freq. Using nominal dt = "
            f"{float(frequency):g} * {simulation_dt:g} = {sample_dt:g} s. "
            "Extra segment-boundary frames have no timestamps in this export; "
            "exact physical timing cannot be recovered.",
            UserWarning,
        )
    else:
        time_source = "explicit_sample_dt"
    if not np.isfinite(sample_dt) or sample_dt <= 0:
        raise ValueError("sample_dt must be positive finite seconds")
    base = np.asarray(
        DEFAULT_BASE_WXYZ if base_wxyz is None else base_wxyz, dtype=float
    )
    if base.shape != (4,) or not np.isfinite(base).all() or np.linalg.norm(base) < 1e-8:
        raise ValueError("Base orientation must be a finite nonzero wxyz quaternion")
    text = group["instructions"][()]
    if isinstance(text, bytes):
        text = text.decode("utf8")
    try:
        instructions = json.loads(text)
    except (ValueError, TypeError):
        instructions = text
    if isinstance(instructions, list):
        instructions = next(
            (x for x in instructions if isinstance(x, str) and x.strip()), ""
        )
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("Native episode has no nonempty instruction")
    meta = dict(
        source="robotwin_native_right_v1",
        pose_order="xyz_wxyz",
        source_pose_frame="world",
        base_wxyz=base.tolist(),
        gripper="continuous commanded width in [0,1], 1 open",
        time_source=time_source,
        sample_dt=sample_dt,
        exported_frequency=float(frequency),
        simulation_dt=simulation_dt,
        arm="right",
    )
    return dict(
        images=images,
        ee_poses=poses[:, None],
        gripper=width[:, None],
        base_wxyz=base,
        instruction=instructions,
        action_type="ee_pose",
        action_frame="base",
        sample_dt=sample_dt,
        gripper_open_value=1.0,
        gripper_closed_value=0.0,
        conversion_metadata=meta,
    )
