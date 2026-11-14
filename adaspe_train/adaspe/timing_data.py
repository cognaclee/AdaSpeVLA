"""Timing-stage data config: carries the measured future state window to the timing label.

The model input is unchanged: LeRobot also loads `state` at the 13 action timestamps, frame 0 becomes
the usual observation state, and the whole window travels separately. After OpenPI pads actions to the
model width, the measured pose (state[4:10]) and gripper width (state[0]) are written into the padded
action channels TIMING_CHANNELS, which the plugin's flow/knot heads never read. Episode-end windows
repeat the last frame (LeRobot clamping), i.e. hold the final pose as AdaSpeVLA's pad_episode_end.
"""
import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

from adaspe.model import TIMING_CHANNELS

KEY = "timing_state"


@dataclasses.dataclass(frozen=True)
class SplitStateWindow(_transforms.DataTransformFn):
    """state (T,19) -> state = frame 0, timing_state = (T,7) [x,y,z, euler xyz, width]."""

    def __call__(self, data):
        window = np.asarray(data["state"])
        data["state"] = window[0]
        data[KEY] = np.concatenate([window[:, 4:10], window[:, 0:1]], -1).astype(np.float32)
        return data


@dataclasses.dataclass(frozen=True)
class KeepKey(_transforms.DataTransformFn):
    """Runs `inner` and re-attaches the timing window, which `inner` would drop."""

    inner: _transforms.DataTransformFn

    def __call__(self, data):
        window = data.get(KEY)
        out = self.inner(data)
        if window is not None:
            out[KEY] = window
        return out


@dataclasses.dataclass(frozen=True)
class PackTimingWindow(_transforms.DataTransformFn):
    """Writes the timing window into the zero-padded action channels."""

    def __call__(self, data):
        if KEY not in data:
            return data
        window = data.pop(KEY)
        actions = np.array(data["actions"], copy=True)
        if actions.shape[0] != window.shape[0] or np.any(actions[:, TIMING_CHANNELS] != 0):
            raise ValueError("timing channels must be zero padding of a same-length action chunk")
        actions[:, TIMING_CHANNELS] = window
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoTimingDataConfig(_config.LeRobotLiberoDataConfig):
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        base = super().create(assets_dirs, model_config)
        repack = base.repack_transforms
        (rp,) = repack.inputs
        repack = _transforms.Group(
            inputs=[SplitStateWindow(), _transforms.RepackTransform({**rp.structure, KEY: KEY})],
            outputs=repack.outputs,
        )
        data_tf = _transforms.Group(inputs=[KeepKey(t) for t in base.data_transforms.inputs], outputs=base.data_transforms.outputs)
        model_tf = _transforms.Group(
            inputs=[*base.model_transforms.inputs, PackTimingWindow()], outputs=base.model_transforms.outputs
        )
        return dataclasses.replace(
            base, action_sequence_keys=("actions", "state"),
            repack_transforms=repack, data_transforms=data_tf, model_transforms=model_tf,
        )
