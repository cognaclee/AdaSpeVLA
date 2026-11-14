"""Co-train one AdaSpeVLA policy over multiple RoboEval task variants."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from adaspevla.benchmarks.roboeval_control import TASK_FAMILIES
from adaspevla.bspline.learnable_knots import motion_variation
from adaspevla.datasets.action_contract import calibrate_motion_scales, validate_motion_scales
from adaspevla.datasets.hdf5_common import load_hdf5_dataset
from adaspevla.benchmarks.pose_math import PROPRIO_PER_ARM
from adaspevla.utils.training import train_two_stage, window_options
from scripts.benchmarks.common_args import add_training_arguments


class MultiTaskWindows(Dataset):
    """Concatenate task windows and share calibration across the joint train set."""
    def __init__(self, datasets):
        if len(datasets) < 2:
            raise ValueError('Co-training requires at least two tasks')
        self.datasets = datasets
        self.lengths = [len(ds) for ds in datasets]
        self.ends = np.cumsum(self.lengths).tolist()
        self.samples = [sample for ds in datasets for sample in ds.samples]
        self.num_views = datasets[0].num_views
        self.num_arms = datasets[0].num_arms
        self.per_arm_knots = datasets[0].per_arm_knots
        if any(ds.num_views != self.num_views or ds.per_arm_knots != self.per_arm_knots for ds in datasets):
            raise ValueError('All co-training datasets must share camera and knot layouts')

        motion_mask = np.ones(self.samples[0][1].shape[-1], dtype=bool)
        motion_mask[6::7] = False
        scales = calibrate_motion_scales([s[1][:, motion_mask] for s in self.samples])
        variations=[]
        for sample in self.samples:
            pooled_scales=validate_motion_scales(scales,self.num_arms).double()
            actions=torch.from_numpy(sample[1][:,motion_mask])[None].double()/pooled_scales
            times=torch.from_numpy(np.asarray(sample[2],dtype=np.float64))[None]
            variations.append(motion_variation(actions,times)[0].numpy())
        e_crit=max(float(np.quantile(np.concatenate(variations),.95)),1e-6)
        proprio=np.stack([s[6] for s in self.samples]).reshape(-1,self.num_arms,PROPRIO_PER_ARM)
        mean=np.zeros((self.num_arms,PROPRIO_PER_ARM));std=np.ones_like(mean)
        mean[:,:3]=proprio[...,:3].mean(0)
        std[:,:3]=np.maximum(proprio[...,:3].std(0),1e-3)
        self.motion_scale_values=scales.tolist()
        self.calibration=dict(datasets[0].calibration)
        self.calibration.update(motion_scale_values=self.motion_scale_values,e_crit=e_crit,
            proprio_mean=mean.reshape(-1).tolist(),proprio_std=std.reshape(-1).tolist(),
            joint_training_tasks=[getattr(ds,'task_name','unknown') for ds in datasets],
            calibration_scope='pooled windows across all co-training task variants')
        self.calibration.pop('source_conversion',None)
        family_counts={}
        for ds in datasets:
            family=TASK_FAMILIES[ds.task_name]
            family_counts[family]=family_counts.get(family,0)+len(ds)
        weights=[]
        for ds in datasets:
            family=TASK_FAMILIES[ds.task_name]
            weights.extend([1.0/family_counts[family]]*len(ds))
        self.sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),
                                            num_samples=len(weights),replacement=True)
        for ds in datasets:
            ds.motion_scale_values=self.motion_scale_values
            ds.motion_scale=validate_motion_scales(scales,ds.num_arms)
            ds.e_crit=e_crit
            ds.calibration=self.calibration

    @property
    def include_diagnostic_targets(self):
        return all(ds.include_diagnostic_targets for ds in self.datasets)

    @include_diagnostic_targets.setter
    def include_diagnostic_targets(self,value):
        for ds in self.datasets:ds.include_diagnostic_targets=value

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        task_index=bisect_right(self.ends,index)
        start=0 if task_index==0 else self.ends[task_index-1]
        return self.datasets[task_index][index-start]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir',required=True)
    p.add_argument('--task_names',nargs='+',required=True,choices=sorted(TASK_FAMILIES))
    add_training_arguments(p,'./checkpoints/roboeval_cotrain')
    a=p.parse_args()
    if len(set(a.task_names))!=len(a.task_names):raise ValueError('Duplicate task variants')
    options=window_options(a);options['num_views']=3
    datasets=[]
    for task in a.task_names:
        manifest=json.loads((Path(a.data_dir)/task/'manifest.json').read_text())
        if manifest.get('task')!=task:raise ValueError(f'{task}: manifest task mismatch')
        ds=load_hdf5_dataset(a.data_dir,task,14,a.chunk_horizon,a.num_control_points,a.degree,'roboeval',dataset_options=options,lazy_images=True)
        ds.task_name=task;datasets.append(ds)
        if len(manifest['episodes'])<1:raise ValueError(f'{task}: no retained demonstrations')
        print(f'Loaded {task}: {len(ds)} windows from {len(manifest["episodes"])} episodes',flush=True)
    joint=MultiTaskWindows(datasets)
    print(f'Joint set: {len(joint)} windows from {len(datasets)} task variants',flush=True)
    a.task_name='RoboEvalAll8Tasks'
    train_two_stage(a,joint,'roboeval',14)


if __name__=='__main__':main()
