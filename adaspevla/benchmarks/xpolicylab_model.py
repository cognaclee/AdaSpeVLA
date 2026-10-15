"""XPolicyLab policy adapter: AdaSpeVLA on RoboTwin 2.0 and RoboDojo.

XPolicyLab asks the model for one action chunk per observation and executes the
whole chunk before observing again. AdaSpeVLA's chunk is the executor prefix: its
length is ``max(1, min(floor(H/dt), ceil(tau/dt)))`` control steps, where ``tau``
comes from the timing head (``query_mode=adaptive``) or a fixed baseline interval
(``query_mode=fixed``). Each predicted base-frame offset is relative to the
end-effector pose observed at query time and is mapped to an absolute
``*_ee_pose`` target by Eq. delta_to_target, because both benchmarks' ``ee``
action mode plans to a pose target per step. The gripper width is sent as a
continuous command in [0,1] (``binary_gripper=true`` thresholds it at 0.5).

One executed action corresponds to one recorded demonstration sample, so the
control period defaults to the conversion's ``sample_dt``. RoboTwin's ``ee`` mode
moves to each target with a motion planner and does not enforce that period;
reported times are therefore nominal (see docs/benchmarks.md).
"""

import atexit
import os
from pathlib import Path

import numpy as np
import torch

from .pose_math import (
    base_rotation,
    delta_to_target,
    normalized_pose,
    pose_error,
    proprio_features,
)
from .rollout import EpisodeRecorder, PrefixPlanner, append_jsonl, load_policy
from .xpolicylab_data import ARM_PREFIX

try:  # The template only adds error helpers; keep the adapter importable without it.
    from XPolicyLab.model_template import ModelTemplate
except ImportError:  # pragma: no cover - exercised only outside XPolicyLab
    ModelTemplate = object


def deployment_profile(calibration):
    """Arms, cameras, base rotations and sample period recorded at conversion."""
    meta = calibration.get("source_conversion")
    if not meta:
        raise ValueError(
            "Checkpoint has no source_conversion metadata. Train on data converted with "
            "scripts.benchmarks.convert_xpolicylab (or --robotwin_native_right)."
        )
    if meta.get("source") == "robotwin_native_right_v1":
        return dict(
            arms=["right"],
            cameras=["cam_head", "cam_right_wrist"],
            base_wxyz=[meta["base_wxyz"]],
            sample_dt=float(meta["sample_dt"]),
        )
    if meta.get("source") == "xpolicylab_ee_pose_v1":
        return dict(
            arms=list(meta["arms"]),
            cameras=list(meta["cameras"]),
            base_wxyz=[list(q) for q in meta["base_wxyz"]],
            sample_dt=float(meta["sample_dt"]),
        )
    raise ValueError(f"Unsupported conversion source {meta.get('source')!r}")


def _env_override(cfg, key, cast=str):
    value = os.environ.get(f"ADASPEVLA_{key.upper()}")
    if value is not None and value != "":
        return cast(value)
    return cfg.get(key)


def resolve_checkpoint(cfg, policy_dir):
    explicit = _env_override(cfg, "checkpoint_path")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    name = cfg.get("ckpt_name")
    if name:
        path = Path(str(name))
        candidates.append(path if path.is_absolute() else Path(policy_dir) / path)
        setting = "-".join(
            str(cfg.get(k))
            for k in ("bench_name", "ckpt_name", "env_cfg_type", "action_type", "seed")
        )
        candidates.append(Path(policy_dir) / "checkpoints" / setting)
        candidates.append(Path(policy_dir) / "checkpoints" / setting.rsplit("-", 1)[0])
    for path in candidates:
        if path.is_file():
            return path
        if path.is_dir():
            latest = sorted(path.glob("*_latest.pt"))
            if len(latest) == 1:
                return latest[0]
            if len(latest) > 1:
                raise ValueError(f"Several *_latest.pt checkpoints in {path}; set checkpoint_path")
    raise FileNotFoundError(
        "No AdaSpeVLA checkpoint found; tried " + ", ".join(str(p) for p in candidates)
    )


class _EnvState:
    def __init__(self, planner, recorder):
        self.planner, self.recorder = planner, recorder
        self.obs = None
        self.pending = []
        self.instruction = None
        self.last_widths = None

    def reset(self):
        self.planner.reset()
        self.recorder.reset()
        self.obs = None
        self.pending = []
        self.instruction = None
        self.last_widths = None


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.cfg = dict(model_cfg)
        self.policy_dir = Path(self.cfg.get("policy_dir") or Path(__file__).resolve().parent)
        if str(self.cfg.get("action_type", "ee")).lower() not in ("ee", "endpose"):
            raise ValueError(
                "AdaSpeVLA emits anchor-relative offsets mapped to end-effector poses; "
                "evaluate with action_type=ee"
            )
        device = self.cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.ckpt_path = resolve_checkpoint(self.cfg, self.policy_dir)
        self.policy, self.calibration, _ = load_policy(self.ckpt_path, self.device)
        profile = deployment_profile(self.calibration)
        self.arms, self.cameras = profile["arms"], profile["cameras"]
        if self.policy.num_arms != len(self.arms) or self.policy.config["num_views"] != len(
            self.cameras
        ):
            raise ValueError("Checkpoint arm/camera counts disagree with its conversion metadata")
        self.world_from_base = [base_rotation(q) for q in profile["base_wxyz"]]
        control_dt = _env_override(self.cfg, "control_dt", float)
        self.control_dt = float(control_dt or profile["sample_dt"])
        self.query_mode = _env_override(self.cfg, "query_mode") or "adaptive"
        interval = _env_override(self.cfg, "fixed_query_interval", float)
        self.fixed_query_interval = None if interval is None else float(interval)
        binary = _env_override(self.cfg, "binary_gripper")
        self.binary_gripper = str(binary).lower() in ("1", "true", "yes")
        steps = _env_override(self.cfg, "num_ode_steps", int)
        self.num_ode_steps = None if steps is None else int(steps)
        self.debug_env = os.environ.get("EVAL_ENV_TYPE", "") == "debug"
        metrics = _env_override(self.cfg, "metrics_dir")
        self.knot_mode = (
            "per_arm" if self.policy.config.get("per_arm_knots") else "shared"
        )
        tag = (
            "adaptive"
            if self.query_mode == "adaptive"
            else f"fixed_{self.fixed_query_interval:g}s"
        )
        if self.knot_mode == "per_arm":
            tag += "-per_arm_knots"
        self.metrics_path = (
            Path(metrics) if metrics else self.policy_dir / "eval_metrics"
        ) / f"{self.cfg.get('bench_name')}-{self.cfg.get('task_name')}-{tag}.jsonl"
        self.case = {}
        self.states = {}
        # Clients that never send trial_end/reset after the last episode still get it saved.
        atexit.register(self.reset)
        print(
            f"[AdaSpeVLA] {self.ckpt_path} arms={self.arms} cameras={self.cameras} "
            f"control_dt={self.control_dt:g}s query_mode={self.query_mode} "
            f"knots={self.knot_mode} "
            f"fixed_interval={self.fixed_query_interval} metrics={self.metrics_path}"
        )

    # ---------------------------------------------------------------- helpers
    def _state(self, env_idx):
        key = int(env_idx or 0)
        if key not in self.states:
            planner = PrefixPlanner(
                self.policy,
                self.control_dt,
                num_ode_steps=self.num_ode_steps,
                query_mode=self.query_mode,
                fixed_query_interval=self.fixed_query_interval,
            )
            self.states[key] = _EnvState(planner, EpisodeRecorder(self.control_dt, len(self.arms)))
        return self.states[key]

    def _pose(self, obs, prefix):
        key = f"{prefix}ee_pose"
        if key not in obs.get("state", {}):
            raise KeyError(f"Observation lacks state/{key}; enable end-effector state")
        pose = np.asarray(obs["state"][key], dtype=np.float64).reshape(-1)
        if self.debug_env:  # The XPolicyLab debug client sends placeholder poses.
            q = pose[3:] if np.linalg.norm(pose[3:]) > 1e-8 else np.array([1.0, 0, 0, 0])
            return np.concatenate((pose[:3], q / np.linalg.norm(q)))
        return normalized_pose(pose)

    def _images(self, obs):
        frames = []
        for cam in self.cameras:
            try:
                color = obs["vision"][cam]["color"]
            except KeyError as exc:
                raise KeyError(f"Observation lacks vision/{cam}/color") from exc
            color = np.array(color, dtype=np.uint8, copy=True)
            if color.ndim != 3 or color.shape[-1] != 3:
                raise ValueError(f"{cam}: expected decoded (H,W,3) RGB")
            frames.append(color)
        if any(f.shape != frames[0].shape for f in frames):
            raise ValueError("Camera resolutions differ; training required matching views")
        images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255
        return images.unsqueeze(0).to(self.device)

    def _robot_prefixes(self, obs):
        state = obs.get("state", {})
        if "ee_pose" in state:
            return [""]
        return [p for p in ("left_", "right_") if f"{p}ee_pose" in state]

    def _record(self, env, success=None):
        # The command whose result was never observed may have run; later ones did not.
        env.recorder.drop_unexecuted(len(env.pending) - 1)
        if env.recorder.prefixes:
            record = env.recorder.summary(
                success=success,
                bench_name=self.cfg.get("bench_name"),
                task_name=self.case.get("task_name", self.cfg.get("task_name")),
                seed=self.case.get("seed"),
                instruction=self.case.get("instruction") or env.instruction,
                query_mode=self.query_mode,
                fixed_query_interval=self.fixed_query_interval,
                knot_mode=self.knot_mode,
                checkpoint=str(self.ckpt_path),
            )
            append_jsonl(self.metrics_path, record)
        env.reset()

    # ------------------------------------------------------------ XPolicyLab
    def prepare_case(self, case_meta=None):
        self.reset()  # Flush an unreported episode under its own case metadata.
        self.case = dict(case_meta or {})

    def update_obs(self, obs):
        env = self._state(obs.get("env_idx", 0))
        if env.pending:
            # Each observation after a chunk starts reflects the next executed command.
            targets = env.pending.pop(0)
            for prefix, target in targets.items():
                translation, rotation = pose_error(target, self._pose(obs, prefix))
                env.recorder.add_tracking(translation, rotation)
        env.obs = obs

    def update_obs_batch(self, obs_list):
        for obs in obs_list:
            self.update_obs(obs)

    def _plan(self, env):
        obs = env.obs
        if obs is None:
            raise RuntimeError("get_action called before update_obs")
        instruction = obs.get("instruction") or (obs.get("instructions") or [""])[0]
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else ""
        if not instruction:
            instruction = self.case.get("instruction") or str(self.cfg.get("task_name"))
        env.instruction = instruction
        query = {"images": self._images(obs), "instructions": [instruction]}
        if getattr(self.policy, "proprio_dim", 0):
            # Training state: the anchor pose and the previous commanded width (1 open).
            if env.last_widths is None:
                env.last_widths = np.ones(len(self.arms))
            poses = np.stack([self._pose(obs, ARM_PREFIX[a]) for a in self.arms])
            state = proprio_features(poses, env.last_widths, self.world_from_base)
            query["proprio"] = torch.from_numpy(state)[None].to(self.device)
        return env.planner.plan(query)

    def _commands(self, env, actions, info):
        """Map prefix offsets to absolute pose targets from the query-time anchor."""
        obs = env.obs
        robot = self._robot_prefixes(obs)
        prefixes = [ARM_PREFIX[a] for a in self.arms]
        missing = [p for p in prefixes if p not in robot]
        if missing:
            raise KeyError(f"Robot observation lacks arms {missing}; available {robot}")
        anchors = {p: self._pose(obs, p) for p in robot}
        holds = {
            p: np.asarray(obs["state"].get(f"{p}ee_joint_state", [1.0]), np.float32).reshape(-1)
            for p in robot
            if p not in prefixes
        }
        arm_actions = np.asarray(actions).reshape(len(actions), len(self.arms), 7)
        # Eq. delta_to_target: every step of the prefix shares the anchor pose.
        poses = np.stack(
            [
                delta_to_target(anchors[prefix], arm_actions[:, i, :6], self.world_from_base[i])
                for i, prefix in enumerate(prefixes)
            ],
            axis=1,
        )
        env.recorder.add_prefix(actions, info, poses)
        chunk = []
        for step, targets in zip(arm_actions, poses):
            command, pending = {}, {}
            for i, prefix in enumerate(prefixes):
                width = float(np.clip(step[i, 6], 0.0, 1.0))
                if self.binary_gripper:
                    width = 1.0 if width >= 0.5 else 0.0
                pending[prefix] = targets[i]
                command[f"{prefix}ee_pose"] = targets[i].astype(np.float32)
                command[f"{prefix}ee_joint_state"] = np.array([width], dtype=np.float32)
            env.last_widths = np.array(
                [float(command[f"{p}ee_joint_state"][0]) for p in prefixes]
            )
            for prefix, grip in holds.items():  # Unused arms hold their pose.
                command[f"{prefix}ee_pose"] = anchors[prefix].astype(np.float32)
                command[f"{prefix}ee_joint_state"] = grip
            chunk.append(command)
            env.pending.append(pending)
        return chunk

    def get_action(self):
        env = self._state(0)
        env.pending = []
        actions, info = self._plan(env)
        return self._commands(env, actions, info)

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = sorted(self.states)
        envs = [self._state(i) for i in env_idx_list]
        plans = [self._plan(env) for env in envs]
        # Batched clients step every env with the same chunk index, so all envs
        # execute the shortest prefix. Use unbatched evaluation for timing results.
        shortest = min(len(actions) for actions, _ in plans)
        chunks = []
        for env, (actions, info) in zip(envs, plans):
            env.pending = []
            chunks.append(
                self._commands(env, actions[:shortest], dict(info, prefix_steps=shortest))
            )
        return chunks

    def reset(self):
        for env in self.states.values():
            self._record(env, success=None)

    def on_trial_end(self, result=None):
        result = dict(result or {})
        env = self._state(0)
        self._record(env, success=result.get("success"))
        return {"metrics_path": str(self.metrics_path)}
