"""AdaSpeVLA OpenPI configs, registered without editing upstream openpi.

Everything except the model (AdaSpeVLA plugin instead of the pi0.5 action expert) uses the
baseline's public settings or OpenPI 215abfb defaults. Knot-density targets and timing labels are
computed as in AdaSpeVLA: measured pose offsets / motion scales and measured gripper width.
"""
import dataclasses
import os
import re
import sys

import flax.traverse_util
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openpi.models.model as _openpi_model  # noqa: E402  (was `_model`, which the config helper used to shadow)
import openpi.models.pi0_config as pi0_config  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.weight_loaders as weight_loaders  # noqa: E402

from adaspe.model import AdaSpePi05Config  # noqa: E402
from adaspe.timing_data import LeRobotLiberoTimingDataConfig  # noqa: E402
import tasks  # noqa: E402

ROOT = tasks.ROOT
TASK = tasks.TASK  # from the TASK env var (default "bread"); used in config names
REPO_ID = tasks.SPEC["repo_id"]
# Official gs://openpi-assets/checkpoints/pi05_base/params (sizes checked against the bucket listing).
PI05_BASE = f"{ROOT}/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
ASSETS = f"{ROOT}/adaspe_train/assets"
CKPT = f"{ROOT}/checkpoints/adaspe"
# Knot / timing labels as AdaSpeVLA franka v1.x: measured anchor offsets / motion scales and measured
# gripper width (calibrate_timing_anchor.py on this task's data; values live in tasks.py).
TIMING_MOTION_SCALE = tasks.SPEC["motion_scale"]
TIMING_E_CRIT = tasks.SPEC["e_crit"]
if TIMING_MOTION_SCALE is None or TIMING_E_CRIT is None:  # not calibrated yet: only the ref / norm-stats config works
    TIMING_MOTION_SCALE, TIMING_E_CRIT = (1.0,) * 6, 1.0
    _UNCALIBRATED = True
else:
    _UNCALIBRATED = False
ACTION_STEPS = 40_000  # LR schedule unchanged (decay 30k), so the 20k checkpoint equals a 20k-only run
ACTION_EVAL_STEPS = (20_000, 30_000, 40_000)
TIMING_STEPS = 10_000


def _ckpt_step(step: int) -> int:
    """OpenPI saves the final step as num_train_steps-1 (40k -> 39999)."""
    return min(step, ACTION_STEPS - 1)


@dataclasses.dataclass(frozen=True)
class PluginWeightLoader(weight_loaders.WeightLoader):
    """pi05_base PaliGemma weights; plugin heads keep their seed-0 init; pi0.5 action expert is not loaded."""

    params_path: str

    def load(self, params):
        loaded = _openpi_model.restore_params(self.params_path, restore_type=np.ndarray)
        # Plugin layer lists have int keys, so match on "/"-joined strings but keep the model's tuple keys.
        name = lambda k: "/".join(map(str, k))  # noqa: E731
        ref = flax.traverse_util.flatten_dict(params)
        got = {name(k): v for k, v in flax.traverse_util.flatten_dict(loaded).items()}
        missing_backbone = [name(k) for k in ref if name(k).startswith("PaliGemma/") and name(k) not in got]
        if missing_backbone:
            raise ValueError(f"pi05_base lacks backbone params: {missing_backbone[:5]}")
        out = {}
        for k, spec in ref.items():
            v = got.get(name(k))
            out[k] = spec if v is None else (v.astype(spec.dtype) if v.dtype != spec.dtype else v)  # spec keeps init
        return flax.traverse_util.unflatten_dict(out)


@dataclasses.dataclass(frozen=True)
class PluginCheckpointWeightLoader(weight_loaders.WeightLoader):
    """Like OpenPI's CheckpointWeightLoader (all params from an action checkpoint), but handles the plugin's
    int layer keys, which the upstream "/"-joining merge cannot. Every param must be present."""

    params_path: str

    def load(self, params):
        loaded = _openpi_model.restore_params(self.params_path, restore_type=np.ndarray)
        name = lambda k: "/".join(map(str, k))  # noqa: E731
        ref = flax.traverse_util.flatten_dict(params)
        got = {name(k): v for k, v in flax.traverse_util.flatten_dict(loaded).items()}
        missing = [name(k) for k in ref if name(k) not in got]
        if missing:
            raise ValueError(f"{self.params_path} lacks params: {missing[:5]}")
        out = {k: got[name(k)].astype(spec.dtype) if got[name(k)].dtype != spec.dtype else got[name(k)]
               for k, spec in ref.items()}
        return flax.traverse_util.unflatten_dict(out)


def _data():
    # Same keys/transforms as OpenPI's LeRobot Libero template; prompt is the dataset task string.
    return _config.LeRobotLiberoDataConfig(
        repo_id=REPO_ID,
        base_config=_config.DataConfig(prompt_from_task=True),
        assets=_config.AssetsConfig(assets_dir=f"{ASSETS}/baseline", asset_id=REPO_ID),
    )


_COMMON = dict(
    data=_data(),
    batch_size=32,
    seed=0,
    assets_base_dir=ASSETS,
    checkpoint_base_dir=CKPT,
    wandb_enabled=False,
    save_interval=1000,
    keep_period=None,  # each OpenPI train-state checkpoint is ~50 GB; keep only the latest (final kept at end)
)

# Baseline data/model reference (used to recompute norm stats and compare with the baseline repo).
BASELINE_REF = _config.TrainConfig(
    name=f"pi05_{TASK}_franka_ref",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10),
    data=_config.LeRobotLiberoDataConfig(repo_id=REPO_ID, base_config=_config.DataConfig(prompt_from_task=True)),
    batch_size=32,
    assets_base_dir=ASSETS,
    checkpoint_base_dir=CKPT,
    wandb_enabled=False,
)


def _timing_data():
    # _data() plus the measured 13-frame state window used for knot / timing labels (model input unchanged).
    return LeRobotLiberoTimingDataConfig(
        repo_id=REPO_ID,
        base_config=_config.DataConfig(prompt_from_task=True),
        assets=_config.AssetsConfig(assets_dir=f"{ASSETS}/baseline", asset_id=REPO_ID),
    )


def model_config(stage, e_crit=None, motion_scale=None):
    """The plugin config with this recipe's knobs (tasks.py); calibrate_timing_anchor.py passes its own values."""
    return AdaSpePi05Config(
        action_horizon=tasks.ACTION_HORIZON, num_control_points=tasks.NUM_CONTROL_POINTS,
        control_dt=tasks.CONTROL_DT, min_query_interval=tasks.MIN_QUERY_INTERVAL,
        max_query_interval=tasks.MAX_QUERY_INTERVAL, stage=stage,
        e_crit=TIMING_E_CRIT if e_crit is None else e_crit,
        timing_motion_scale=TIMING_MOTION_SCALE if motion_scale is None else tuple(motion_scale),
        timing_label_mode=tasks.LABEL_MODE, knot_label_mode=tasks.LABEL_MODE, knot_motion_norm=tasks.KNOT_MOTION_NORM,
    )



ACTION = _config.TrainConfig(
    name=f"{TASK}_adaspe_action",
    model=model_config("action"),
    weight_loader=PluginWeightLoader(PI05_BASE),
    num_train_steps=ACTION_STEPS,
    **{**_COMMON, "data": _timing_data(), "keep_period": 10_000},  # keep 10k/20k/30k/40k for evaluation
)

# Same model with the plain data config; only tests/check_timing_labels.py uses it, to confirm the extra
# state window leaves the model inputs unchanged.
INPUT_REF = dataclasses.replace(ACTION, name=f"{TASK}_adaspe_input_ref", data=_data())


def timing_config(action_step: int) -> _config.TrainConfig:
    """Timing head only, from the action stage's EMA params at `action_step`."""
    return _config.TrainConfig(
        name=f"{TASK}_adaspe_timing_from{action_step // 1000}k",
        model=model_config("timing"),
        weight_loader=PluginCheckpointWeightLoader(
            f"{CKPT}/{TASK}_adaspe_action/formal/{_ckpt_step(action_step)}/params"),
        freeze_filter=AdaSpePi05Config(stage="timing").get_freeze_filter(),
        num_train_steps=TIMING_STEPS,
        **{**_COMMON, "data": _timing_data()},
    )


TIMINGS = [timing_config(s) for s in ACTION_EVAL_STEPS]

CONFIGS = {c.name: c for c in [BASELINE_REF, ACTION, INPUT_REF, *TIMINGS]}


def register():
    # Until tasks.py has this task's calibration, only the norm-stats reference config is usable.
    for name, cfg in ({BASELINE_REF.name: BASELINE_REF} if _UNCALIBRATED else CONFIGS).items():
        _config._CONFIGS_DICT[name] = cfg
