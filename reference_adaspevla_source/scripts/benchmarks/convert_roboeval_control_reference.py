"""Current official simulator + explicit local control-reference conversion.

No simulator patches. Preserve original demo duration. Retain only trajectories
that succeed both when generated and when the recorded teacher is replayed.
Images, observed state and executed targets come from the same final rollout.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation as R
from roboeval.demonstrations.demo import Demo
from roboeval.demonstrations.demo_converter import DemoConverter
from adaspevla.benchmarks.roboeval_control import CAMERAS, make_env, native_action, observed_poses, offset_targets
from adaspevla.benchmarks.pose_math import anchor_deltas


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def targets_from_joints(robot, joint):
    fk = robot.forward_kinematics(joint[:-2]).reshape(2, 6)
    targets = np.empty((2, 7))
    targets[:, :3] = fk[:, :3]
    rot = R.from_rotvec(fk[:, 3:])
    if robot._pelvis is not None:
        targets[:, :3] += robot._pelvis.get_position()
        q = np.asarray(robot._pelvis.get_quaternion())
        rot = rot * R.from_quat(q[[1, 2, 3, 0]])
    targets[:, 3:] = rot.as_quat()[:, [3, 0, 1, 2]]
    return targets


def teacher(anchor, target, width):
    action = np.empty((2, 7))
    for arm in range(2):
        action[arm, :6] = anchor_deltas(target[arm:arm+1], anchor[arm], R.identity())[0]
        action[arm, 6] = width[arm]
    return action.reshape(14)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--official-repo', required=True)
    p.add_argument('--uuid-manifest', help='Official successful_demos_ee_delta_20hz.txt')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    commit = subprocess.check_output(['git','-C',args.official_repo,'rev-parse','HEAD'],text=True).strip()
    assert commit == '4f16f33595fade301e33b252a1ee665164edf180'
    assert not subprocess.check_output(['git','-C',args.official_repo,'diff','HEAD'],text=True)
    out = Path(args.output) / args.task
    out.mkdir(parents=True, exist_ok=True)
    (out/'raw').mkdir(exist_ok=True)
    config = {'task':args.task,'conversion':'current_ctrl_reference_v1','official_commit':commit,
              'control_frequency':20,'limit':args.limit,'archive_sha256':hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),
              'converter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'control_bridge_sha256':hashlib.sha256(Path(__import__('adaspevla.benchmarks.roboeval_control',fromlist=['__file__']).__file__).read_bytes()).hexdigest(),
              'uuid_manifest_sha256':hashlib.sha256(Path(args.uuid_manifest).read_bytes()).hexdigest() if args.uuid_manifest else None,
              'official_simulator_modified':False,'historical_dataset_equivalence':False}
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text()) == config, 'Refusing incompatible resume'
    write_json(out/'config.json',config)
    status = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {'episodes':{}}
    status['status']='running'
    sim, rgb = make_env(args.task, images=False), make_env(args.task, images=True)
    try:
        with zipfile.ZipFile(args.archive) as archive:
            prefix=f'BimanualPanda/{args.task}/JointPositionActionMode_floating_absolute_joint/lightweight/'
            members=sorted(n for n in archive.namelist() if n.startswith(prefix) and n.endswith('.safetensors'))
            if args.uuid_manifest:
                selected=[]; current=None
                for line in Path(args.uuid_manifest).read_text().splitlines():
                    match=re.match(r'\[(.+?)\]',line)
                    if match: current=match.group(1)
                    elif current==args.task and re.fullmatch(r'[0-9a-f]{32}',line.strip()):
                        selected.append(line.strip())
                expected={Path(member).stem:member for member in members}
                missing=sorted(set(selected)-set(expected))
                assert not missing, f'Official UUIDs missing from archive: {missing[:5]}'
                members=[expected[uid] for uid in selected]
            if args.limit: members=members[:args.limit]
            assert members, 'No matching source demonstrations'
            status['source_count']=len(members)
            for member in members:
                uid=Path(member).stem
                if uid in status['episodes']: continue
                path=out/'raw'/(uid+'.safetensors')
                if not path.exists(): path.write_bytes(archive.read(member))
                raw=Demo.from_safetensors(path)
                demo=DemoConverter.decimate(raw,20,500,robot=raw.metadata.get_robot())
                sim.reset(seed=raw.seed)
                labels=[];generated=False
                for step in demo.timesteps:
                    anchor=observed_poses(sim.robot)
                    target=targets_from_joints(sim.robot,step.executed_action)
                    width=1-np.asarray(step.executed_action[-2:])
                    action=native_action(sim.robot,target,width)
                    assert sim.action_space.contains(action)
                    labels.append(teacher(anchor,target,width))
                    sim.step(action,fast=True)
                    generated=bool(sim.success) or generated
                row={'uuid':uid,'seed':int(raw.seed),'source_frames':len(demo.timesteps),
                     'generation_success':generated,'teacher_replay_success':False,'retained':False}
                if generated:
                    obs,_=rgb.reset(seed=raw.seed)
                    images=[];states=[];widths=[];targets=[];commands=[];success=False;first_success=None
                    for index,label in enumerate(labels):
                        anchor=observed_poses(rgb.robot)
                        images.append(np.stack([obs['rgb_'+c].transpose(1,2,0) for c in CAMERAS]))
                        states.append(anchor);widths.append(rgb.robot.qpos_grippers.copy())
                        target=offset_targets(anchor,label[None])[0]
                        command=label.reshape(2,7)[:,6]
                        action=native_action(rgb.robot,target,command)
                        assert rgb.action_space.contains(action)
                        targets.append(target);commands.append(command.copy())
                        obs,_,_,_,_=rgb.step(action)
                        if rgb.success:
                            success=True
                            if first_success is None:first_success=index+1
                    row['teacher_replay_success']=success
                    row['first_success_step']=first_success
                    if success:
                        images.append(np.stack([obs['rgb_'+c].transpose(1,2,0) for c in CAMERAS]))
                        states.append(observed_poses(rgb.robot));widths.append(rgb.robot.qpos_grippers.copy())
                        states=np.asarray(states);widths=np.asarray(widths)
                        poses=np.concatenate([states[:1],np.asarray(targets)])
                        commands=np.concatenate([widths[:1],np.asarray(commands)])
                        np.savez_compressed(out/(uid+'.npz'), images=np.asarray(images),
                            observed_ee_poses=states,observed_gripper=widths,
                            ee_poses=poses,gripper=commands,timestamps=np.arange(len(states))*.05)
                        row.update(retained=True,frames=len(states))
                status['episodes'][uid]=row
                status['processed']=len(status['episodes'])
                status['retained']=sum(x['retained'] for x in status['episodes'].values())
                write_json(out/'status.json',status)
                print(args.task,status['processed'],'/',len(members),'retained',status['retained'],uid,flush=True)
        status['status']='complete' if status['retained'] else 'failed_no_retained_data'
        write_json(out/'status.json',status)
        assert status['retained'] > 0
    except Exception as exc:
        status['status']='failed';status['error']=repr(exc)
        write_json(out/'status.json',status)
        raise
    finally:
        sim.close();rgb.close()


if __name__=='__main__':main()
