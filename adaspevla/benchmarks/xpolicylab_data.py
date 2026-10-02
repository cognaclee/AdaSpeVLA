"""Convert XPolicyLab v1.0 trajectory HDF5 (RoboTwin 2.0, RoboDojo) to the contract.

The XPolicyLab exporter stores each row's current end-effector pose under
``state/<arm>_ee_poses`` and the next row's pose under ``action/<arm>_ee_poses``
(``[x,y,z,qw,qx,qy,qz]``), with commanded grippers under
``action/<arm>_ee_joint_states`` in [0, 1] (1 open). This module extends the
verified right-arm conversion in ``datasets.robotwin_native`` to any arm subset
and writes already-converted episodes that the existing loaders read unchanged:
``images`` (T, V, H, W, 3) uint8, ``actions`` (T, 7*arms) base-frame twists plus
binary grippers, and explicit action metadata attributes.

Frame consistency matters more than the absolute base orientation: evaluation
integrates predicted twists with the same per-arm ``world_from_base`` recorded in
the conversion metadata, so a wrong but consistent base quaternion only rotates
the learned channels. The metadata also records the timing source.
"""

import json
import warnings
from pathlib import Path

import h5py
import numpy as np

from ..datasets.hdf5_common import read_rgb_camera
from ..datasets.robotwin_native import DEFAULT_BASE_WXYZ
from .pose_math import base_rotation, pose_pairs_to_twists, wxyz_to_rotation

CONVERSION_FILE = "adaspevla_conversion.json"

# Frequency field semantics differ between collectors:
#  - RoboTwin 2.0 stores its save stride (sim steps per saved frame), see
#    docs/robotwin_native.md; nominal dt = stride * simulation_dt.
#  - RoboDojo stores collect_freq in Hz (env/observation_manager/obs_manager.py).
PROFILES = {
    "robotwin": dict(
        frequency_semantics="stride",
        simulation_dt=1 / 250,
        base_wxyz=DEFAULT_BASE_WXYZ,
    ),
    "robodojo": dict(frequency_semantics="hz", simulation_dt=None, base_wxyz=None),
}

ARM_PREFIX = {"left": "left_", "right": "right_", "single": ""}
DEFAULT_CAMERAS = {
    ("right",): ["cam_head", "cam_right_wrist"],
    ("left",): ["cam_head", "cam_left_wrist"],
    ("left", "right"): ["cam_head", "cam_left_wrist", "cam_right_wrist"],
    ("single",): ["cam_head", "cam_wrist"],
}


def default_cameras(arms):
    try:
        return list(DEFAULT_CAMERAS[tuple(arms)])
    except KeyError as exc:
        raise ValueError(f"No default camera set for arms {arms}; pass cameras") from exc


def _read(group, path):
    if path not in group:
        raise ValueError(f"{group.file.filename}: XPolicyLab export requires {path}")
    return np.asarray(group[path])


def _instruction(group):
    text = _read(group, "instructions")[()]
    if isinstance(text, bytes):
        text = text.decode("utf8")
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = text
    if isinstance(parsed, list):
        parsed = next((x for x in parsed if isinstance(x, str) and x.strip()), "")
    if not isinstance(parsed, str) or not parsed.strip():
        raise ValueError(f"{group.file.filename}: no nonempty instruction")
    return parsed


def episode_sample_dt(group, profile, sample_dt=None, simulation_dt=None):
    """Return (sample_dt, time_source) following the profile's frequency field."""
    if sample_dt is not None:
        if not np.isfinite(sample_dt) or sample_dt <= 0:
            raise ValueError("sample_dt must be positive finite seconds")
        return float(sample_dt), "explicit_sample_dt"
    frequency = _read(group, "additional_info/frequency")
    if frequency.ndim != 0 or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("additional_info/frequency must be a positive scalar")
    settings = PROFILES[profile]
    if settings["frequency_semantics"] == "hz":
        return 1.0 / float(frequency), "collector_hz"
    simulation_dt = settings["simulation_dt"] if simulation_dt is None else simulation_dt
    if not np.isfinite(simulation_dt) or simulation_dt <= 0:
        raise ValueError("simulation_dt must be positive")
    return float(frequency) * simulation_dt, "nominal_collector_stride"


def convert_episode(
    group,
    arms=("right",),
    cameras=None,
    profile="robotwin",
    base_wxyz=None,
    sample_dt=None,
    simulation_dt=None,
    gripper_threshold=0.5,
):
    """Convert one XPolicyLab episode group; returns contract fields and metadata."""
    arms = list(arms)
    if not arms or len(set(arms)) != len(arms) or any(a not in ARM_PREFIX for a in arms):
        raise ValueError("arms must be distinct values from left/right/single")
    if "single" in arms and len(arms) != 1:
        raise ValueError("A single-arm robot cannot be combined with left/right arms")
    if len(arms) > 2:
        raise ValueError("At most two arms are supported by the 7/14-channel contract")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    if not 0 < gripper_threshold < 1:
        raise ValueError("gripper_threshold must lie strictly between 0 and 1")
    cameras = default_cameras(arms) if cameras is None else list(cameras)
    dt, time_source = episode_sample_dt(group, profile, sample_dt, simulation_dt)
    if base_wxyz is None:
        base_wxyz = PROFILES[profile]["base_wxyz"]
    if base_wxyz is None:
        raise ValueError(
            f"Profile {profile!r} has no verified base orientation; pass base_wxyz "
            "(use [1,0,0,0] when the exported poses are already base-relative)"
        )
    base_wxyz = np.asarray(base_wxyz, dtype=float)
    if base_wxyz.shape == (4,):
        base_wxyz = np.repeat(base_wxyz[None], len(arms), axis=0)
    if base_wxyz.shape != (len(arms), 4):
        raise ValueError("base_wxyz must be one quaternion or one per arm")

    views = [read_rgb_camera(group, f"vision/{c}/colors", robotwin_codec=True) for c in cameras]
    if any(v.shape != views[0].shape for v in views):
        raise ValueError("Selected cameras must have matching frame counts and resolution")
    images = np.stack(views, axis=1)

    channels = []
    for arm, base in zip(arms, base_wxyz):
        prefix = ARM_PREFIX[arm]
        current = _read(group, f"state/{prefix}ee_poses").astype(np.float64)
        target = _read(group, f"action/{prefix}ee_poses").astype(np.float64)
        if current.ndim != 2 or current.shape[1] != 7 or target.shape != current.shape:
            raise ValueError(f"Expected matching (T,7) {arm} state/action poses")
        if len(current) < 2 or not np.isfinite(current).all() or not np.isfinite(target).all():
            raise ValueError(f"{arm} poses must be finite with at least two rows")
        r0 = wxyz_to_rotation(current[:, 3:], unit_atol=1e-3)
        r1 = wxyz_to_rotation(target[:, 3:], unit_atol=1e-3)
        if (
            not np.allclose(target[:-1, :3], current[1:, :3], atol=1e-5)
            or np.max((r1[:-1] * r0[1:].inv()).magnitude()) > 1e-4
        ):
            raise ValueError(
                f"{arm} state/action poses do not have the one-frame shift; refusing to infer alignment"
            )
        twist = pose_pairs_to_twists(current, target, dt, base_rotation(base))
        grip = _read(group, f"action/{prefix}ee_joint_states")
        if grip.shape != (len(current), 1) or not np.isfinite(grip).all():
            raise ValueError(f"{arm} gripper must have shape (T,1)")
        if (grip < -1e-6).any() or (grip > 1 + 1e-6).any():
            raise ValueError(f"{arm} gripper values must be normalized to [0,1] (1 open)")
        channels.append(
            np.concatenate((twist, (grip >= gripper_threshold).astype(float)), axis=-1)
        )
    if len(images) != len(channels[0]):
        raise ValueError("Image and pose sample counts differ")
    meta = dict(
        source="xpolicylab_ee_pose_v1",
        profile=profile,
        arms=arms,
        cameras=cameras,
        pose_order="xyz_wxyz",
        source_pose_frame="exported_ee_pose_frame",
        base_wxyz=base_wxyz.tolist(),
        gripper_threshold=gripper_threshold,
        gripper_open_value=1.0,
        gripper_closed_value=0.0,
        time_source=time_source,
        sample_dt=dt,
    )
    return dict(
        images=images,
        actions=np.concatenate(channels, axis=-1).astype(np.float32),
        instruction=_instruction(group),
        sample_dt=dt,
        meta=meta,
    )


def write_contract_episode(path, episode, task_name):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "images", data=episode["images"], compression="gzip", compression_opts=4,
            chunks=(1,) + episode["images"].shape[1:],
        )
        f["actions"] = episode["actions"]
        f.attrs.update(
            task_name=task_name,
            instruction=episode["instruction"],
            action_type="twist",
            action_frame="base",
            gripper_open_value=float(episode["meta"]["gripper_open_value"]),
            gripper_closed_value=float(episode["meta"]["gripper_closed_value"]),
            adaspevla_conversion=json.dumps(episode["meta"], sort_keys=True),
        )
        if "timestamps" in episode:
            f["timestamps"] = episode["timestamps"]
        else:
            f.attrs["sample_dt"] = float(episode["sample_dt"])


def find_episode_files(source_dir, max_episodes=None):
    files = sorted(
        set(Path(source_dir).rglob("*.hdf5")) | set(Path(source_dir).rglob("*.h5"))
    )
    if not files:
        raise FileNotFoundError(f"No HDF5 episodes under {source_dir}")
    return files[:max_episodes] if max_episodes else files


def convert_directory(source_dir, output_dir, task_name, max_episodes=None, **options):
    """Convert every episode file below ``source_dir`` into ``output_dir/task_name``."""
    out = Path(output_dir) / task_name
    metas, count = [], 0
    for path in find_episode_files(source_dir, max_episodes):
        with h5py.File(path, "r") as root:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                episode = convert_episode(root, **options)
        write_contract_episode(out / f"episode_{count:05d}.hdf5", episode, task_name)
        metas.append(episode["meta"])
        count += 1
    if any(m != metas[0] for m in metas):
        raise ValueError("Conversion conventions differ across episodes (timing or frames)")
    meta = dict(metas[0], task_name=task_name, episodes=count, source_dir=str(source_dir))
    (out / CONVERSION_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True))
    if meta["time_source"] == "nominal_collector_stride":
        warnings.warn(
            "RoboTwin frequency stores a save stride; using nominal dt = "
            f"{meta['sample_dt']:g} s. Exact physical timing cannot be recovered.",
            UserWarning,
        )
    return meta


def read_conversion(data_dir, task_name):
    path = Path(data_dir) / task_name / CONVERSION_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; convert data with scripts.benchmarks.convert_xpolicylab first"
        )
    return json.loads(path.read_text())
