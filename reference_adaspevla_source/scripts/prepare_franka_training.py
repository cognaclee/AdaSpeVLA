"""Audit all converted Franka files and materialize joint training calibration."""
import argparse
import json
from pathlib import Path
import hashlib

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from adaspevla.datasets.franka_lerobot import digest
from adaspevla.datasets.franka_training import load_franka_training, policy_config
from adaspevla.datasets.bspline_dataset import collate_windows
from adaspevla.benchmarks.pose_math import anchor_deltas, delta_to_target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    recipe = json.loads(a.recipe.read_text())
    rows = []
    identity = Rotation.identity()
    for task in recipe['task_names']:
        directory = a.data_dir / task
        manifest = json.loads((directory/'manifest.json').read_text())
        count = 0
        error = 0.
        clipped = 0
        raw_lo, raw_hi = float('inf'), -float('inf')
        for record in manifest['episodes']:
            path = directory/record['file']
            if digest(path) != record['sha256']:
                raise ValueError(f'Converted file hash mismatch: {path}')
            with h5py.File(path, 'r') as h:
                assert h.attrs['action_frame'] == 'episode_reset'
                assert h.attrs['supervision'] == 'measured_pose'
                pose = h['ee_poses'][:,0]
                state = h['raw/state'][:]
                times = h['timestamps'][:]
                widths = h['gripper'][:,0]
                assert np.all(np.diff(times)>0) and np.isfinite(pose).all()
                assert np.all((widths>=0)&(widths<=1))
                np.testing.assert_array_equal(pose[:,:3], state[:,4:7])
                np.testing.assert_allclose(Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix(),
                                           Rotation.from_euler('xyz', state[:,7:10]).as_matrix(), atol=1e-12)
                assert h['images'].shape == (len(pose),2,128,128,3)
                assert h['raw/actions'].shape == (len(pose),7)
                np.testing.assert_allclose(widths, np.clip(state[:,0],0,1), atol=0,rtol=0)
                for anchor in [0,len(pose)//2,len(pose)-1]:
                    offsets = anchor_deltas(pose,pose[anchor],identity)
                    restored = delta_to_target(pose[anchor],offsets,identity)
                    # Quaternion signs may differ; compare rotations geometrically.
                    pe = np.max(np.abs(restored[:,:3]-pose[:,:3]))
                    re = (Rotation.from_quat(restored[:,[4,5,6,3]]).inv()*Rotation.from_quat(pose[:,[4,5,6,3]])).magnitude().max()
                    error = max(error,float(pe),float(re))
                assert error < 1e-9
                clipped += int(h.attrs['gripper_clipped_samples'])
                raw_lo,raw_hi = min(raw_lo,float(state[:,0].min())),max(raw_hi,float(state[:,0].max()))
                count += len(pose)
        assert count == manifest['total_frames']
        rows.append(dict(task=task, episodes=len(manifest['episodes']),frames=count,
                         max_pose_roundtrip_error=error, gripper_clipped_samples=clipped,
                         raw_gripper_range=[raw_lo,raw_hi]))
        print(rows[-1],flush=True)
    joint, signature = load_franka_training(a.data_dir, recipe['task_names'],recipe)
    expected = sum(r['frames']-r['episodes'] for r in rows)
    assert len(joint) == expected
    weight_sums = []
    start = 0
    for end in joint.ends:
        weight_sums.append(float(joint.sampler.weights[start:end].sum()))
        start = end
    np.testing.assert_allclose(weight_sums, np.ones(len(rows)),atol=1e-12)
    starts = [0]+joint.ends[:-1]
    sample_indices = []
    for start,end in zip(starts,joint.ends):
        sample_indices.extend(np.linspace(start,end-1,min(20,end-start),dtype=int).tolist())
    for index in sample_indices:
        sample = joint[index]
        for key,value in sample.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite target {key}, window {index}')
    batch = collate_windows([joint[i] for i in starts])
    report = dict(status='passed', source='provided collector + action/state definitions',
                  data_dir=str(a.data_dir.resolve()),data_manifest_sha256=signature,
                  recipe_sha256=hashlib.sha256(a.recipe.read_bytes()).hexdigest(),
                  tasks=rows, total_episodes=sum(x['episodes'] for x in rows),
                  total_frames=sum(x['frames'] for x in rows),total_windows=len(joint),
                  sampled_windows_checked=len(sample_indices),uniform_task_probabilities=[1/len(rows)]*len(rows),
                  batch_shapes={k:list(v.shape) for k,v in batch.items() if torch.is_tensor(v)},
                  model_config=policy_config(joint,recipe),calibration=joint.calibration,
                  split='all episodes; no held-out performance claimed',
                  limitations=['No physical robot replay','No full-batch GPU VRAM/throughput test','Exported 10Hz timestamps do not measure collection wall-time'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(f'PASSED: {len(joint)} windows; report {a.output}',flush=True)


if __name__ == '__main__':
    main()
