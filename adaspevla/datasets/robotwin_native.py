"""Right-arm adapter for RoboTwin's native XPolicyLab v1.0 export.

Conventions verified against RoboTwin ea8b21121ebb3cd201ff5b3fe361944ac94eda3f:
  envs/utils/pkl2hdf5.py, envs/robot/robot.py, envs/_base_task.py.
See docs/robotwin_native.md for source links and timing limitations.
"""

import json
import warnings
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_BASE_WXYZ = [0.707, 0.0, 0.0, 0.707]


def native_right_episode(
    group,
    images,
    sample_dt=None,
    base_wxyz=None,
    gripper_threshold=0.5,
    simulation_dt=1 / 250,
):
    """Convert consecutive observed poses to average physical base-frame twists.

    This is trajectory finite differencing, not reconstruction of the simulator's
    original joint-controller command. Gripper binarization is explicit.
    """

    def read(path):
        if path not in group:
            raise ValueError(f"Native RoboTwin export requires {path}")
        return np.asarray(group[path])

    current = read("state/right_ee_poses").astype(np.float64)
    target = read("action/right_ee_poses").astype(np.float64)
    if (
        current.ndim != 2
        or current.shape[1] != 7
        or target.shape != current.shape
        or len(current) < 2
    ):
        raise ValueError("Expected matching (T,7) right-arm state/action poses")
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("Right-arm poses must be finite")

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
            "State/action poses do not have the official one-frame shift; refusing to infer alignment"
        )
    if len(images) != len(current):
        raise ValueError("Image and right-arm pose lengths differ")
    frequency = read("additional_info/frequency")
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
    world_to_base = Rotation.from_quat(base[[1, 2, 3, 0]]).inv()
    velocity = world_to_base.apply((target[:, :3] - current[:, :3]) / sample_dt)
    angular = world_to_base.apply((r1 * r0.inv()).as_rotvec() / sample_dt)
    grip = read("action/right_ee_joint_states")
    if (
        grip.shape != (len(current), 1)
        or not np.isfinite(grip).all()
        or (grip < -1e-6).any()
        or (grip > 1 + 1e-6).any()
    ):
        raise ValueError(
            "Native right gripper must have shape (T,1) and values in [0,1]"
        )
    if not 0 < gripper_threshold < 1:
        raise ValueError("gripper_threshold must lie strictly between 0 and 1")
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
        gripper_threshold=gripper_threshold,
        time_source=time_source,
        sample_dt=sample_dt,
        exported_frequency=float(frequency),
        simulation_dt=simulation_dt,
        arm="right",
    )
    return dict(
        images=images,
        actions=np.concatenate(
            (velocity, angular, (grip >= gripper_threshold).astype(float)), axis=-1
        ).astype(np.float32),
        instruction=instructions,
        action_type="twist",
        action_frame="base",
        sample_dt=sample_dt,
        gripper_open_value=1.0,
        gripper_closed_value=0.0,
        conversion_metadata=meta,
    )
