"""RoboCasa365 (PandaOmron, robosuite OSC) conversion and closed-loop evaluation.

Data. RoboCasa365 ships LeRobot v2.1 datasets (lerobot==0.3.3). Its 12-D action is
a controller-normalized OSC *setpoint offset*, not a physical motion, so it is not
used as the motion target. Instead the achieved end-effector pose relative to the
robot base (``observation.state[7:10]`` position, ``[10:14]`` xyzw quaternion, see
robocasa/models/assets/groot_dataset_assets/PandaOmron_modality.json) is stored
with the recorded ``timestamp`` column as the pose contract; training windows
convert it to anchor-relative offsets (Eq. anchor_delta). This dataset logs only
a binary gripper command (``gripper_close``, robosuite: -1 open, +1 closed), so
the width is mapped to {0, 1}; the width at t_j is the command issued at row
j-1, matching the pose that command reaches.

The policy does not move the mobile base, so episodes whose ``base_motion`` or
``control_mode`` channels are active are rejected by default.

Control. In arm mode robosuite's OSC sets ``goal = achieved + scale * clip(a)``
in the base frame (robosuite/controllers/parts/arm/osc.py, ``goal_update_mode``
"achieved"), with ``goal_ori = R(scale * a_rot) @ achieved_ori``. Commanding
``a = (target - current) / scale`` therefore tracks the spline pose targets
(Eq. delta_to_target from the pose observed at query time) with fresh state
feedback at every 20 Hz control step, while the policy itself is queried only at
the adaptive prefix boundaries. The binary gripper receives 1[w >= 0.5].
"""

import json
import warnings
from pathlib import Path

import numpy as np

from .pose_math import delta_to_target, pose_error, proprio_features, wxyz_to_rotation
from .rollout import EpisodeRecorder, PrefixPlanner
from scipy.spatial.transform import Rotation

STATE_EEF_POS = slice(7, 10)
STATE_EEF_QUAT_XYZW = slice(10, 14)
ACTION_BASE_MOTION = slice(0, 4)
ACTION_CONTROL_MODE = 4
ACTION_GRIPPER_CLOSE = 11
DEFAULT_CAMERAS = ["robot0_agentview_left", "robot0_eye_in_hand"]
# default_pandaomron.json: OSC_POSE output_max for position (m) and rotation (rad).
OSC_POSITION_SCALE = 0.05
OSC_ROTATION_SCALE = 0.5
IDENTITY = Rotation.identity()


def _xyzw_to_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    return quat[..., [3, 0, 1, 2]]


def _column(df, name):
    if name not in df.columns:
        raise ValueError(f"LeRobot episode lacks column {name!r}; columns={list(df.columns)}")
    return np.stack([np.asarray(v, dtype=np.float64) for v in df[name]])


def decode_video(path):
    """Decode every frame of an MP4 to RGB uint8 (T,H,W,3)."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames)


class LeRobotV21:
    """Minimal reader for a local LeRobot v2.1 dataset directory."""

    def __init__(self, root):
        self.root = Path(root)
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"{info_path} not found; expected a LeRobot v2.1 dataset")
        self.info = json.loads(info_path.read_text())
        version = str(self.info.get("codebase_version", ""))
        if not version.startswith("v2"):
            raise ValueError(f"Only LeRobot v2.x layouts are supported, got {version!r}")
        self.tasks = {}
        tasks_path = self.root / "meta" / "tasks.jsonl"
        if tasks_path.is_file():
            for line in tasks_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.tasks[int(row["task_index"])] = row["task"]
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.data_path = self.info.get(
            "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        self.video_path = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )

    def episode_indices(self):
        files = sorted(self.root.glob("data/chunk-*/episode_*.parquet"))
        return [int(f.stem.split("_")[-1]) for f in files]

    def frames(self, episode_index):
        import pandas as pd

        chunk = episode_index // self.chunks_size
        path = self.root / self.data_path.format(episode_chunk=chunk, episode_index=episode_index)
        return pd.read_parquet(path)

    def video(self, episode_index, camera):
        chunk = episode_index // self.chunks_size
        key = f"observation.images.{camera}"
        path = self.root / self.video_path.format(
            episode_chunk=chunk, video_key=key, episode_index=episode_index
        )
        return decode_video(path)


def episode_trajectory(df, episode_index, base_motion="error"):
    """Base-frame poses (T,7 wxyz), binary open widths (T,), times and base flag.

    Returns None when ``base_motion='drop'`` and the episode moves the base.
    """
    state = _column(df, "observation.state")
    action = _column(df, "action")
    times = np.asarray(df["timestamp"], dtype=np.float64)
    if state.shape[1] < 14 or action.shape[1] != 12:
        raise ValueError("Expected PandaOmron 16-D state and 12-D action")
    if len(times) < 3 or (np.diff(times) <= 0).any():
        raise ValueError(f"Episode {episode_index}: timestamps must strictly increase")
    moved = (np.abs(action[:, ACTION_BASE_MOTION]) > 1e-6).any() or (
        action[:, ACTION_CONTROL_MODE] > 0.5
    ).any()
    if moved and base_motion != "ignore":
        if base_motion == "drop":
            return None
        raise ValueError(
            f"Episode {episode_index} moves the mobile base, which AdaSpeVLA does not control. "
            "Choose a stationary task, or pass --base_motion drop/ignore explicitly."
        )
    pose = np.concatenate(
        (state[:, STATE_EEF_POS], _xyzw_to_wxyz(state[:, STATE_EEF_QUAT_XYZW])), axis=-1
    )
    pose[:, 3:] /= np.linalg.norm(pose[:, 3:], axis=-1, keepdims=True)
    # Binary log: width 1 open / 0 closed; the width at t_j is row j-1's command.
    grip_cmd = action[:, ACTION_GRIPPER_CLOSE]
    gripper_open = (grip_cmd <= 0).astype(np.float64)
    gripper_open = np.concatenate((gripper_open[:1], gripper_open[:-1]))
    return pose, gripper_open, times, bool(moved)


def convert_episode(dataset, episode_index, cameras=None, base_motion="error"):
    """Return contract fields for one RoboCasa episode, or None if it is dropped."""
    cameras = list(cameras or DEFAULT_CAMERAS)
    df = dataset.frames(episode_index)
    trajectory = episode_trajectory(df, episode_index, base_motion)
    if trajectory is None:
        return None
    pose, gripper_open, times, moved = trajectory
    views = [dataset.video(episode_index, c) for c in cameras]
    for camera, view in zip(cameras, views):
        if len(view) != len(times):
            raise ValueError(
                f"Episode {episode_index}: {camera} has {len(view)} frames for {len(times)} rows"
            )
    if any(v.shape[1:] != views[0].shape[1:] for v in views):
        raise ValueError("Selected cameras must share one resolution")
    images = np.stack(views, axis=1)
    task_index = int(np.asarray(df["task_index"])[0]) if "task_index" in df.columns else None
    instruction = dataset.tasks.get(task_index, "") if task_index is not None else ""
    if not instruction.strip():
        raise ValueError(f"Episode {episode_index}: no task instruction in meta/tasks.jsonl")
    dt = np.diff(times)
    meta = dict(
        source="robocasa365_lerobot_v21",
        arms=["right"],
        cameras=cameras,
        pose_source="observation.state base_to_eef pos[7:10], quat_xyzw[10:14]",
        pose_frame="robot0_base",
        base_wxyz=[[1.0, 0.0, 0.0, 0.0]],
        time_source="lerobot_timestamps",
        sample_dt=float(np.median(dt)),
        gripper_source="action.gripper_close (<=0 open), previous row, binary log",
        gripper_open_value=1.0,
        gripper_closed_value=0.0,
        osc_position_scale=OSC_POSITION_SCALE,
        osc_rotation_scale=OSC_ROTATION_SCALE,
        base_motion_policy=base_motion,
        image_shape=list(images.shape[2:]),
    )
    return dict(
        images=images,
        ee_poses=pose[:, None],
        gripper=gripper_open[:, None].astype(np.float32),
        base_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        timestamps=(times - times[0]).astype(np.float64),
        instruction=instruction,
        meta=meta,
        base_moved=bool(moved),
    )


def convert_dataset(root, output_dir, task_name, cameras=None, base_motion="error", max_episodes=None):
    from .xpolicylab_data import CONVERSION_FILE, write_contract_episode

    dataset = LeRobotV21(root)
    out = Path(output_dir) / task_name
    indices = dataset.episode_indices()[:max_episodes] if max_episodes else dataset.episode_indices()
    if not indices:
        raise FileNotFoundError(f"No episodes in {root}")
    written = dropped = moved = 0
    sample_dts, meta = [], None
    for index in indices:
        episode = convert_episode(dataset, index, cameras, base_motion)
        if episode is None:
            dropped += 1
            continue
        moved += episode["base_moved"]
        write_contract_episode(out / f"episode_{written:05d}.hdf5", episode, task_name)
        sample_dts.append(episode["meta"]["sample_dt"])
        current = {k: v for k, v in episode["meta"].items() if k != "sample_dt"}
        if meta is not None and current != meta:
            raise ValueError("Conversion conventions differ across episodes")
        meta = current
        written += 1
    if not written:
        raise ValueError("Every episode was dropped for base motion; choose a stationary task")
    meta = dict(
        meta,
        sample_dt=float(np.median(sample_dts)),
        task_name=task_name,
        episodes=written,
        dropped_base_motion_episodes=dropped,
        kept_base_motion_episodes=moved,
        source_dir=str(root),
    )
    (out / CONVERSION_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True))
    if moved:
        warnings.warn(f"{moved} kept episodes move the base; evaluation keeps it still")
    return meta


def osc_delta_command(target_pose, current_pose, position_scale, rotation_scale):
    """Normalized OSC delta that moves the arm-mode goal onto ``target_pose``."""
    position = (np.asarray(target_pose[:3]) - np.asarray(current_pose[:3])) / position_scale
    rotation = (
        wxyz_to_rotation(target_pose[3:]) * wxyz_to_rotation(current_pose[3:]).inv()
    ).as_rotvec() / rotation_scale
    return np.clip(position, -1, 1), np.clip(rotation, -1, 1)


def observed_pose(obs):
    position = np.asarray(obs["state.end_effector_position_relative"], dtype=np.float64)
    quat = _xyzw_to_wxyz(obs["state.end_effector_rotation_relative"])
    return np.concatenate((position.reshape(3), quat.reshape(4) / np.linalg.norm(quat)))


class RoboCasaClosedLoop:
    """Run AdaSpeVLA prefixes in a RoboCasaGymEnv-compatible environment."""

    def __init__(self, policy, meta, query_mode="adaptive", fixed_query_interval=None,
                 num_ode_steps=None, control_dt=None, device="cpu"):
        import torch

        self.torch, self.device = torch, device
        self.cameras = list(meta["cameras"])
        self.control_dt = float(control_dt or meta["sample_dt"])
        self.position_scale = float(meta.get("osc_position_scale", OSC_POSITION_SCALE))
        self.rotation_scale = float(meta.get("osc_rotation_scale", OSC_ROTATION_SCALE))
        self.planner = PrefixPlanner(
            policy, self.control_dt, num_ode_steps=num_ode_steps,
            query_mode=query_mode, fixed_query_interval=fixed_query_interval,
        )
        self.recorder = EpisodeRecorder(self.control_dt, 1)
        self.use_proprio = bool(getattr(policy, "proprio_dim", 0))

    def _images(self, obs):
        frames = [np.array(obs[f"video.{c}"], dtype=np.uint8, copy=True) for c in self.cameras]
        images = self.torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255
        return images.unsqueeze(0).to(self.device)

    def run_episode(self, env, seed, max_steps):
        obs, _ = env.reset(seed=seed)
        self.planner.reset()
        self.recorder.reset()
        success, steps = False, 0
        # Training state uses the previous binary command (1 open); episodes start open.
        last_width = 1.0
        instruction = str(obs.get("annotation.human.task_description", ""))
        while steps < max_steps and not success:
            query = {"images": self._images(obs), "instructions": [instruction]}
            if self.use_proprio:
                state = proprio_features(observed_pose(obs)[None], [last_width], [IDENTITY])
                query["proprio"] = self.torch.from_numpy(state)[None].to(self.device)
            actions, info = self.planner.plan(query)
            # Eq. delta_to_target from the pose observed with this query.
            targets = delta_to_target(observed_pose(obs), actions[:, :6], IDENTITY)
            self.recorder.add_prefix(actions, info, targets[:, None])
            for action, target in zip(actions, targets):
                position, rotation = osc_delta_command(
                    target, observed_pose(obs), self.position_scale, self.rotation_scale
                )
                last_width = 1.0 if action[6] >= 0.5 else 0.0
                obs, _, done, truncated, step_info = env.step({
                    "action.end_effector_position": position,
                    "action.end_effector_rotation": rotation,
                    "action.gripper_close": np.array([1.0 - last_width]),
                    "action.base_motion": np.zeros(4),
                    "action.control_mode": np.array([0.0]),
                })
                steps += 1
                self.recorder.add_tracking(*pose_error(target, observed_pose(obs)))
                success = bool(step_info.get("success", False))
                if success or steps >= max_steps or done or truncated:
                    break
            if done or truncated:
                break
        # Actions planned but not executed after success are not counted.
        self.recorder.drop_unexecuted(len(self.recorder.actions) - steps)
        return self.recorder.summary(success=success, seed=seed, instruction=instruction)
