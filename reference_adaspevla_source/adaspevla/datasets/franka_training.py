"""Task-balanced Franka windows with one shared calibration, matching simulation."""
from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .action_contract import calibrate_motion_scales, validate_motion_scales
from .hdf5_common import load_hdf5_dataset
from .franka_lerobot import validate_mapping
from ..bspline.learnable_knots import motion_variation


class FrankaTaskWindows(Dataset):
    def __init__(self, datasets, manifests):
        if not datasets or any(len(d) == 0 for d in datasets):
            raise ValueError('All selected tasks need nonempty windows')
        self.datasets = datasets
        self.ends = np.cumsum([len(d) for d in datasets]).tolist()
        self.samples = [s for d in datasets for s in d.samples]
        self.num_views, self.num_arms, self.action_dim = 2, 1, 7
        self.per_arm_knots = False
        frames = {d.calibration['action_frame'] for d in datasets}
        if len(frames) != 1:
            raise ValueError('Cannot jointly train incompatible coordinate frames')
        scales = calibrate_motion_scales([s[1][:, :6] for s in self.samples])
        variations = []
        for s in self.samples:
            motion = torch.from_numpy(s[1][:, :6])[None].double() / scales.double()
            times = torch.from_numpy(np.asarray(s[2], dtype=np.float64))[None]
            variations.append(motion_variation(motion, times)[0].numpy())
        e_crit = max(float(np.quantile(np.concatenate(variations), .95)), 1e-6)
        proprio = np.stack([s[6] for s in self.samples])
        mean, std = np.zeros(10), np.ones(10)
        mean[:3] = proprio[:, :3].mean(0)
        std[:3] = np.maximum(proprio[:, :3].std(0), 1e-3)
        self.motion_scale_values = scales.tolist()
        self.calibration = dict(datasets[0].calibration)
        self.calibration.update(motion_scale_values=self.motion_scale_values, e_crit=e_crit,
                                proprio_mean=mean.tolist(), proprio_std=std.tolist(),
                                calibration_scope='all selected real episodes, pooled windows',
                                joint_training_tasks=[m['task'] for m in manifests],
                                task_weights=[1 / len(datasets)] * len(datasets),
                                source_conversion=manifests)
        weights = [1 / len(d) for d in datasets for _ in range(len(d))]
        self.sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                             num_samples=len(weights), replacement=True)
        for d in datasets:
            d.motion_scale_values = self.motion_scale_values
            d.motion_scale = validate_motion_scales(scales, 1)
            d.e_crit = e_crit
            d.calibration = self.calibration

    @property
    def include_diagnostic_targets(self):
        return self.datasets[0].include_diagnostic_targets

    @include_diagnostic_targets.setter
    def include_diagnostic_targets(self, value):
        for d in self.datasets:
            d.include_diagnostic_targets = value

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        task = bisect_right(self.ends, index)
        start = 0 if task == 0 else self.ends[task - 1]
        return self.datasets[task][index - start]


def load_franka_training(data_dir, task_names, recipe):
    datasets, manifests = [], []
    signature = hashlib.sha256()
    for task in task_names:
        directory = Path(data_dir) / task
        payload = (directory / 'manifest.json').read_bytes()
        signature.update(task.encode() + b'\0' + payload)
        meta = json.loads(payload)
        if meta.get('source') != 'franka_lerobot_v21' or not meta.get('complete') or meta.get('task') != task:
            raise ValueError(f'{task}: incomplete or wrong manifest')
        validate_mapping(meta['mapping'])
        expected = {x['file'] for x in meta['episodes']}
        actual = {str(x.relative_to(directory)) for x in directory.rglob('*.h5')} | {str(x.relative_to(directory)) for x in directory.rglob('*.hdf5')}
        if not expected or expected != actual:
            raise ValueError(f'{task}: manifest and files differ')
        ds = load_hdf5_dataset(data_dir, task, 7, recipe['chunk_horizon'],
                              recipe['num_control_points'], recipe['degree'], 'franka',
                              lazy_images=True, dataset_options=dict(num_views=2,
                              control_dt=recipe['control_dt'], pad_episode_end=True,
                              min_query_interval=recipe['min_query_interval'],
                              max_query_interval=recipe['max_query_interval']))
        ds.task_name = task
        datasets.append(ds)
        manifests.append(meta)
        print(f'{task}: {len(meta["episodes"])} episodes, {len(ds)} windows', flush=True)
    joint = FrankaTaskWindows(datasets, manifests)
    return joint, signature.hexdigest()


def policy_config(dataset, recipe, test_backbone=False):
    durations = np.array([s[2][-1] for s in dataset.samples])
    return dict(vlm_name=recipe.get('vlm_name','pi05'), action_dim=7, num_control_points=recipe['num_control_points'],
                degree=recipe['degree'], cond_dim=512, num_views=2, freeze_vlm=recipe.get("freeze_vlm",True),
                knot_loss_coeff=.5, flow_loss_coeff=1., horizon_duration=float(np.median(durations)),
                min_query_interval=recipe['min_query_interval'], max_query_interval=recipe['max_query_interval'],
                min_interval_eps=.01, test_mode=test_backbone,
                flow_hidden_dim=48 if test_backbone else 384, flow_num_layers=1 if test_backbone else 6,
                flow_num_heads=6, motion_scale_values=dataset.motion_scale_values,
                timing_probe_count=16, timing_num_ode_steps=15,
                supported_horizon_range=[float(durations.min()), float(durations.max())],
                spatial_tokens=True, augment=False, vlm_mode='joint', vlm_dtype='auto',
                vlm_layers=recipe.get("vlm_layers",6), image_fit='pad', proprio_dim=10,
                proprio_mean=dataset.calibration['proprio_mean'],
                proprio_std=dataset.calibration['proprio_std'],
                action_frame=dataset.calibration['action_frame'])
