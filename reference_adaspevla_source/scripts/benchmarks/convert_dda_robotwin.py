"""Convert DDA-VLA RoboTwin demonstrations (joint states only) to the pose contract.

DDA-VLA's collector (DDA-VLA/scripts/collect_robotwin_pairs.py, aloha-agilex, demo_clean,
raster renderer) stores per frame the joint-state vector (left 6 joints + gripper, right 6 +
gripper, gripper in [0,1] with 1 open), the next frame's vector as the action, and three views
(head, left wrist, right wrist) squashed from 320x240 to 224x224. This script:

* computes each arm's end-effector pose by forward kinematics in SAPIEN, as RoboTwin's
  ``get_<arm>_ee_pose``: the world pose of joint 6 with its rotation times the embodiment's
  ``global_trans_matrix``. On official XPolicyLab demos this reproduces
  ``state/<arm>_ee_poses`` with a median error of 0 mm (p95 2.9 mm, max 3.6 mm while moving);
* sets the gripper width at row j to the width commanded at row j-1 (row 0 for t0), the
  XPolicyLab convention of ``robotwin_native.xpolicylab_arm_trajectory``;
* resizes the views back to 320x240.

It needs SAPIEN (RoboTwin's environment). Episodes are numbered from ``--start_index`` so they
can sit next to an official conversion of the same task; the conversion metadata is written by
the caller. Note that DDA renders are darker than RoboTwin's default renderer and its wrist views
clip the near gripper fingers.

Example:
    python -m scripts.benchmarks.convert_dda_robotwin --task_name beat_block_hammer \
        --source_root $SCRATCH/dda_vla/robotwin_pairs_x60 --output_dir data/converted/robotwin-d110 \
        --start_index 50 --max_episodes 60
"""

import argparse
import glob
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from adaspevla.benchmarks.xpolicylab_data import write_contract_episode

def default_embodiment_dir():
    workspace = Path(__file__).resolve().parents[3]
    robotwin = Path(os.environ.get("ROBOTWIN_ROOT", workspace / "external/RoboTwin"))
    return robotwin / "assets/embodiments/aloha-agilex"


def default_metadata_root():
    workspace = Path(__file__).resolve().parents[3]
    local = workspace / "test/screening/v3_recipe_screen_20261019/data/converted/robotwin"
    return str(local) if local.is_dir() else "data/converted/robotwin"

ROBOT_POSE = ([0.0, -0.65, 0.0], [0.707, 0.0, 0.0, 0.707])  # config.yml robot_pose (wxyz)
GLOBAL_TRANS = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
ARMS = (("left", "fl", slice(0, 6), 6), ("right", "fr", slice(7, 13), 13))
FRAME = (320, 240)  # RoboTwin demo_clean camera resolution (width, height)


class AlohaKinematics:
    """End-effector poses of both aloha-agilex arms from joint states (SAPIEN, no renderer)."""

    def __init__(self, embodiment_dir):
        import sapien

        urdf = Path(embodiment_dir) / "urdf" / "arx5_description_isaac.urdf"
        text = urdf.read_text()
        # Kinematics only: geometry would need a render device the conversion does not have.
        text = re.sub(r"<visual\b.*?</visual>|<collision\b.*?</collision>", "", text, flags=re.S)
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as handle:
            handle.write(text)
        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])  # the robot lives only as long as its scene
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(handle.name)
        Path(handle.name).unlink()
        self.robot.set_root_pose(sapien.Pose(*ROBOT_POSE))
        names = [j.get_name() for j in self.robot.get_active_joints()]
        self.index = {arm: [names.index(f"{p}_joint{i}") for i in range(1, 7)] for arm, p, _, _ in ARMS}
        self.ee = {arm: self.robot.find_joint_by_name(f"{p}_joint6") for arm, p, _, _ in ARMS}
        self.dof = len(names)

    def poses(self, arm, joints):
        """(T, 6) joint angles -> (T, 7) poses [x, y, z, qw, qx, qy, qz] in the world frame."""
        from transforms3d.quaternions import mat2quat, quat2mat

        out = []
        for q in joints:
            qpos = np.zeros(self.dof)
            qpos[self.index[arm]] = q
            self.robot.set_qpos(qpos)
            pose = self.ee[arm].get_global_pose()
            out.append(np.r_[pose.p, mat2quat(quat2mat(pose.q) @ GLOBAL_TRANS)])
        return np.asarray(out, dtype=np.float64)


def convert_file(path, kinematics, meta):
    data = torch.load(path, map_location="cpu", weights_only=False)
    states, actions = data["states"].double().numpy(), data["actions"].double().numpy()
    images = data["images"].numpy()  # (T, V, 3, H, W) uint8
    if states.shape != actions.shape or states.shape[1] != 14 or len(images) != len(states):
        raise ValueError(f"{path}: expected matching (T,14) states/actions and T image rows")
    if not np.allclose(actions[:-1], states[1:], atol=1e-5):
        raise ValueError(f"{path}: actions are not the next frame's joint states")
    poses, widths = [], []
    for arm, _, joints, grip in ARMS:
        poses.append(kinematics.poses(arm, states[:, joints]))
        commanded = actions[:, grip]
        widths.append(np.r_[commanded[0], commanded[:-1]])
    widths = np.stack(widths, axis=1)
    if (widths < -1e-6).any() or (widths > 1 + 1e-6).any():
        raise ValueError(f"{path}: gripper widths outside [0, 1]")
    frames = np.stack([
        np.stack([
            np.asarray(Image.fromarray(view.transpose(1, 2, 0)).resize(FRAME, Image.Resampling.BILINEAR))
            for view in row
        ])
        for row in images
    ])
    return dict(
        images=frames,
        ee_poses=np.stack(poses, axis=1),
        gripper=widths.clip(0, 1).astype(np.float32),
        base_wxyz=np.asarray(meta["base_wxyz"], dtype=float),
        instruction=str(data["instruction"]),
        sample_dt=float(meta["sample_dt"]),
        meta=dict(meta, source_episode=str(path), seed=int(data["seed"])),
    )


def source_files(source_root, task_name):
    """Train episodes of a DDA collection and its helper collections (<root>_h<k>), main first."""
    files = []
    for root in [source_root] + sorted(glob.glob(f"{source_root}_h*")):
        manifest = Path(root) / task_name / "manifest.json"
        if manifest.is_file():
            episodes = json.loads(manifest.read_text())["episodes"]
            files += [e["path"] for e in episodes if e["split"] == "train"]
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--source_root", required=True, help="DDA collection root, e.g. robotwin_pairs_x60")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_index", type=int, default=0, help="Number of the first written episode")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--meta_from", default=default_metadata_root(),
                        help="Official conversion whose metadata (cameras, base, sample_dt) these match")
    parser.add_argument("--embodiment_dir", default=None)
    args = parser.parse_args()
    meta = json.loads((Path(args.meta_from) / args.task_name / "adaspevla_conversion.json").read_text())
    meta = {k: meta[k] for k in ("arms", "base_wxyz", "cameras", "gripper", "gripper_closed_value",
                                 "gripper_open_value", "pose_order", "profile", "sample_dt", "time_source")}
    meta.update(source="dda_vla_robotwin_joint_states", source_pose_frame="sapien_fk_get_ee_pose")
    embodiment = args.embodiment_dir or str(default_embodiment_dir())
    kinematics = AlohaKinematics(embodiment)
    files = source_files(args.source_root, args.task_name)[: args.max_episodes]
    out = Path(args.output_dir) / args.task_name
    for offset, path in enumerate(files):
        episode = convert_file(path, kinematics, meta)
        write_contract_episode(out / f"episode_{args.start_index + offset:05d}.hdf5", episode, args.task_name)
        print(f"{Path(path).name} -> episode_{args.start_index + offset:05d} ({len(episode['images'])} rows)", flush=True)
    print(f"Converted {len(files)} DDA episodes of {args.task_name}")


if __name__ == "__main__":
    main()
