"""Closed-loop prefix planning, query-timing baselines and episode metrics.

``PrefixPlanner`` drives the unmodified ``SynchronousSplineExecutor`` so that one
policy call yields exactly the prefix the executor would command before its next
query. Benchmarks that consume an action chunk per call (XPolicyLab, RoboCasa
wrappers) can therefore execute AdaSpeVLA's adaptive prefixes without
re-implementing Eq. (steps).

``QueryTimingPolicy`` optionally replaces the predicted interval with a fixed one.
It is an evaluation baseline for the paper's fixed-vs-supervised ablation; the
same checkpoint, knots, control points and grippers are used in both modes.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..execution import SynchronousSplineExecutor
from ..models.adaspe_vla_policy import AdaSpeVLAPolicy
from ..utils.metrics import compute_chunk_boundary_jump, compute_trajectory_jerk


def load_policy(ckpt_path, device="cpu"):
    """Load a completed stage-2 version-3 checkpoint strictly."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("format_version") != 3 or "model_config" not in ckpt:
        raise ValueError(
            "Checkpoint lacks the corrected action/timing contract; retrain with format_version=3"
        )
    if ckpt.get("training_stage") != "timing":
        raise ValueError("Evaluation requires a checkpoint trained through stage 2")
    policy = AdaSpeVLAPolicy(**ckpt["model_config"]).to(device)
    policy.load_state_dict(ckpt["model_state"], strict=True)
    policy.eval()
    return policy, ckpt.get("calibration", {}), ckpt


class QueryTimingPolicy:
    """Record each prediction and optionally impose a fixed query interval."""

    def __init__(self, policy, query_mode="adaptive", fixed_query_interval=None):
        if query_mode not in ("adaptive", "fixed"):
            raise ValueError("query_mode must be adaptive or fixed")
        if query_mode == "fixed" and (
            fixed_query_interval is None
            or not math.isfinite(fixed_query_interval)
            or fixed_query_interval <= 0
        ):
            raise ValueError("Fixed query timing needs a positive interval in seconds")
        self.policy = policy
        self.query_mode = query_mode
        self.fixed_query_interval = fixed_query_interval
        self.last_prediction = None
        self.last_inference_seconds = None

    def __getattr__(self, name):
        return getattr(self.policy, name)

    @torch.no_grad()
    def predict_action_spline(self, *args, **kwargs):
        device = next(self.policy.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = self.policy.predict_action_spline(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.last_inference_seconds = time.perf_counter() - started
        prediction = dict(prediction)
        prediction["predicted_query_interval"] = prediction["query_interval"]
        if self.query_mode == "fixed":
            prediction["query_interval"] = torch.full_like(
                prediction["query_interval"], self.fixed_query_interval
            )
        self.last_prediction = prediction
        return prediction


class PrefixPlanner:
    """Return one executor prefix of physical commands per new observation."""

    def __init__(
        self,
        policy,
        control_dt,
        num_ode_steps=None,
        query_mode="adaptive",
        fixed_query_interval=None,
        boundary_matching=False,
        motion_limits=None,
    ):
        self.timing = QueryTimingPolicy(policy, query_mode, fixed_query_interval)
        self._observation = None
        self.executor = SynchronousSplineExecutor(
            self.timing,
            lambda: self._observation,
            lambda action: None,
            control_dt=control_dt,
            num_ode_steps=(
                policy.timing_num_ode_steps if num_ode_steps is None else num_ode_steps
            ),
            command_type="twist",
            boundary_matching=boundary_matching,
            motion_limits=motion_limits,
        )

    def reset(self):
        self.executor.reset()
        self._observation = None

    def plan(self, observation):
        """Query once and return ``(actions (N, 7*arms), info)`` for the prefix."""
        self._observation = observation
        # Every call follows a fresh observation: start a new prefix even if the
        # benchmark ended the previous one early.
        self.executor.prefix_steps = 0
        actions = []
        action, queried = self.executor.step()
        if not queried:
            raise RuntimeError("Executor did not query on a fresh observation")
        actions.append(action.numpy().copy())
        while self.executor.step_index < self.executor.prefix_steps:
            action, _ = self.executor.step()
            actions.append(action.numpy().copy())
        prediction = self.timing.last_prediction
        info = dict(
            requested_interval=float(prediction["query_interval"].item()),
            predicted_interval=float(prediction["predicted_query_interval"].item()),
            horizon=float(prediction["horizon_duration"].item()),
            prefix_steps=len(actions),
            inference_seconds=float(self.timing.last_inference_seconds),
        )
        return np.stack(actions), info


class EpisodeRecorder:
    """Accumulate per-episode success, query and smoothness statistics."""

    def __init__(self, control_dt, num_arms):
        self.control_dt, self.num_arms = control_dt, num_arms
        self.reset()

    def reset(self):
        self.prefixes = []
        self.actions = []
        self.tracking = []
        self.meta = {}

    def add_prefix(self, actions, info):
        self.prefixes.append(dict(info))
        self.actions.extend(np.asarray(actions))

    def drop_unexecuted(self, count):
        """Forget the last ``count`` planned commands that were never executed."""
        if count > 0:
            self.actions = self.actions[: max(0, len(self.actions) - count)]

    def add_tracking(self, translation, rotation):
        self.tracking.append((float(translation), float(rotation)))

    def summary(self, success=None, **extra):
        steps = len(self.actions)
        queries = len(self.prefixes)
        record = dict(self.meta, **extra)
        record.update(
            success=None if success is None else bool(success),
            executed_steps=steps,
            policy_queries=queries,
            queries_per_step=queries / steps if steps else None,
            control_dt=self.control_dt,
            nominal_execution_seconds=steps * self.control_dt,
            nominal_query_hz=(queries / (steps * self.control_dt)) if steps else None,
        )
        if queries:
            infer = np.array([p["inference_seconds"] for p in self.prefixes])
            requested = np.array([p["requested_interval"] for p in self.prefixes])
            predicted = np.array([p["predicted_interval"] for p in self.prefixes])
            prefix = np.array([p["prefix_steps"] for p in self.prefixes])
            record.update(
                inference_ms_mean=float(1e3 * infer.mean()),
                inference_ms_total=float(1e3 * infer.sum()),
                requested_interval_mean=float(requested.mean()),
                predicted_interval_mean=float(predicted.mean()),
                predicted_interval_min=float(predicted.min()),
                predicted_interval_max=float(predicted.max()),
                prefix_steps_mean=float(prefix.mean()),
                prefix_steps_hist={
                    str(k): int(v) for k, v in zip(*np.unique(prefix, return_counts=True))
                },
            )
        if steps:
            arms = np.asarray(self.actions).reshape(steps, self.num_arms, 7)
            record["linear_jerk"] = compute_trajectory_jerk(
                arms[..., :3].reshape(steps, -1), self.control_dt, "velocity"
            )
            record["angular_jerk"] = compute_trajectory_jerk(
                arms[..., 3:6].reshape(steps, -1), self.control_dt, "velocity"
            )
            jumps = []
            start = 0
            for p in self.prefixes:
                end = min(start + p["prefix_steps"], steps)
                if start >= 3 and end - start >= 3:
                    jumps.append(
                        compute_chunk_boundary_jump(
                            arms[start - 3 : start, :, :6].reshape(3, -1),
                            arms[start:end, :, :6].reshape(end - start, -1),
                            self.control_dt,
                            self.control_dt,
                        )["c0_value_jump"]
                    )
                start = end
            if jumps:
                record["boundary_c0_jump_mean"] = float(np.mean(jumps))
        if self.tracking:
            tracking = np.array(self.tracking)
            record.update(
                tracking_translation_m_mean=float(tracking[:, 0].mean()),
                tracking_rotation_rad_mean=float(tracking[:, 1].mean()),
            )
        return record


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def summarize_records(records):
    """Aggregate episode records grouped by (task, knot mode, query mode, interval)."""
    groups = {}
    for record in records:
        key = (
            record.get("task_name"),
            record.get("knot_mode", "shared"),
            record.get("query_mode"),
            record.get("fixed_query_interval"),
        )
        groups.setdefault(key, []).append(record)
    rows = []
    for (task, knot_mode, mode, interval), items in sorted(
        groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        def mean(name):
            values = [r[name] for r in items if r.get(name) is not None]
            return float(np.mean(values)) if values else None

        judged = [r["success"] for r in items if r.get("success") is not None]
        rows.append(
            dict(
                task_name=task,
                knot_mode=knot_mode,
                query_mode=mode,
                fixed_query_interval=interval,
                episodes=len(items),
                success_rate=float(np.mean(judged)) if judged else None,
                queries_per_step=mean("queries_per_step"),
                nominal_query_hz=mean("nominal_query_hz"),
                inference_ms_mean=mean("inference_ms_mean"),
                inference_ms_total=mean("inference_ms_total"),
                executed_steps=mean("executed_steps"),
                predicted_interval_mean=mean("predicted_interval_mean"),
                tracking_translation_m_mean=mean("tracking_translation_m_mean"),
                tracking_rotation_rad_mean=mean("tracking_rotation_rad_mean"),
                linear_jerk=mean("linear_jerk"),
                angular_jerk=mean("angular_jerk"),
                boundary_c0_jump_mean=mean("boundary_c0_jump_mean"),
            )
        )
    return rows
