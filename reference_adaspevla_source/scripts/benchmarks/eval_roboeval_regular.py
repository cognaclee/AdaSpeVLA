"""Recorded regular-version rollouts; official comparison requires a locked protocol.

Diagnostic mode deliberately makes no leaderboard-comparability claim. Keep native
wall-clock completion_time and simulated time separate instead of overwriting CT.
"""
import argparse
import hashlib
import json
import math
import subprocess
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from adaspevla.benchmarks.roboeval_control import (
    CAMERAS, INSTRUCTIONS, make_env, native_action, observed_poses, offset_targets,
    absolute_action, proprio_poses, reset_observation,
)

RAW_KEYS = {
    'success': 'success', 'tp': 'subtask_progress',
    'bavd': 'bimanual_arm_velocity_difference',
    'bgvd': 'bimanual_gripper_vertical_difference',
    'cpl': 'total_cartesian_path_length', 'ct': 'completion_time',
    'ecc': 'env_collision_count', 'jpl': 'total_joint_path_length',
    'mcj': 'overall_avg_cartesian_jerk', 'mjj': 'overall_avg_joint_jerk',
    'opl': 'total_orientation_path_length', 'scc': 'self_collision_count',
    'sc': 'slip_count',
}
REQUIRED_OFFICIAL_CHECKS = (
    'combined_variation', 'mujoco_version', 'evaluation_seeds', 'episode_budget',
    'episode_count', 'environment_commit', 'training_data_split', 'metric_aggregation',
)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Non-finite metric')
    return value


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False))
    temp.replace(path)


class Video:
    def __init__(self, path):
        self.process = subprocess.Popen([
            '/usr/bin/ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '768x256', '-r', '20', '-i', '-',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path),
        ], stdin=subprocess.PIPE)

    def frame(self, obs):
        frame = np.concatenate([obs['rgb_' + c].transpose(1, 2, 0) for c in CAMERAS], axis=1)
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait(timeout=30):
            raise RuntimeError('Video encoding failed')


def protocol_for(args):
    protocol = json.loads(Path(args.protocol).read_text())
    if args.task not in protocol['tasks']:
        raise ValueError('Task is not in the protocol')
    if not args.diagnostic:
        missing = [k for k in REQUIRED_OFFICIAL_CHECKS
                   if protocol.get('official_checks', {}).get(k, {}).get('status') != 'verified']
        local_protocol = protocol.get('protocol_kind') == 'adaspevla_cotrain_rho_inspired'
        if local_protocol:
            if protocol.get('status') != 'locked' or not protocol.get('training_data_sha256'):
                raise ValueError('Local co-training protocol/data manifest not locked')
            if len(protocol['evaluation_seeds']) != 100 or any(v != 250 for v in protocol['max_steps'].values()):
                raise ValueError('Local co-training protocol must use 100 seeds and 250-step limits')
        elif not protocol.get('official_checks') or missing or protocol.get('status') != 'locked':
            raise ValueError('Official comparison is not locked; unresolved: ' + ', '.join(missing))
        gate = json.loads(Path(protocol['validation_gate']).read_text())
        if gate.get('status') != 'passed' or gate.get('protocol_sha256') != hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest():
            raise ValueError('Small validation has not passed for this exact protocol')
    return protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--socket', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--diagnostic', action='store_true')
    a = p.parse_args()
    protocol = protocol_for(a)
    seeds = protocol['diagnostic_seeds'] if a.diagnostic else protocol['evaluation_seeds']
    limit = protocol['diagnostic_max_steps'] if a.diagnostic else protocol['max_steps'][a.task]
    if not seeds or len(seeds) != len(set(seeds)) or limit <= 0:
        raise ValueError('Invalid seeds or budget')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    import adaspevla.benchmarks.roboeval_control as bridge
    source_hashes[str(Path(bridge.__file__).resolve())] = hashlib.sha256(Path(bridge.__file__).read_bytes()).hexdigest()
    config = {'task': a.task, 'protocol': protocol, 'diagnostic': a.diagnostic,
              'comparison_status': ('diagnostic_only' if a.diagnostic else
                                    'local_rho_inspired_protocol' if protocol.get('protocol_kind') == 'adaspevla_cotrain_rho_inspired'
                                    else 'official_protocol_aligned'),
              'source_hashes': source_hashes}
    config_path = out / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Refusing to mix runs with different configurations')
    write_json(config_path, config)
    absolute = protocol.get('controller_mode') == 'absolute_ee'
    env = make_env(a.task, absolute=absolute)
    conn = None
    records = []
    try:
        conn = Client(a.socket, family='AF_UNIX')
        conn.send({'op': 'metadata'})
        metadata = plain(conn.recv())
        metadata_path = out / 'policy_metadata.json'
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
            raise ValueError('Refusing to mix checkpoints in one evaluation directory')
        write_json(metadata_path, metadata)
        for seed in seeds:
            path = out / f'episode_{seed}.json'
            if path.exists():
                records.append(json.loads(path.read_text()))
                continue
            start = time.time()
            obs, _ = reset_observation(env,seed)
            initial_qpos = env.robot.qpos.copy()
            # Full simulator state is audit-only and never sent to the policy.
            initial_scene_qpos = env._mojo.data.qpos.copy()
            initial_scene_qvel = env._mojo.data.qvel.copy()
            initial_body_pos = env._mojo.model.body_pos.copy()
            initial_body_quat = env._mojo.model.body_quat.copy()
            np.savez_compressed(out / f'initial_rgb_{seed}.npz',
                                images=np.stack([obs['rgb_'+c] for c in CAMERAS]))
            if env.success:
                raise RuntimeError('Environment reports success at reset')
            conn.send({'op': 'reset', 'seed': seed})
            assert conn.recv()['ok']
            actions, queries = [], []
            clipped = 0
            success = terminated = truncated = False
            video_path = out / f'episode_{seed}.mp4'
            video = Video(video_path)
            try:
                video.frame(obs)
                while len(actions) < limit and not (success or terminated or truncated):
                    anchor = proprio_poses(env.robot, obs) if absolute else observed_poses(env.robot)
                    conn.send({'op': 'predict', 'images': np.stack([
                        obs['rgb_' + c].transpose(1, 2, 0) for c in CAMERAS]),
                        'poses': anchor, 'gripper': obs['proprioception_grippers'],
                        'instruction': INSTRUCTIONS[a.task]})
                    response = conn.recv()
                    if 'error' in response:
                        raise RuntimeError(response['error'])
                    prefix = np.asarray(response['actions'])
                    if len(prefix) == 0 or not np.isfinite(prefix).all():
                        raise ValueError('Invalid action prefix')
                    queries.append(response['info'])
                    for planned, world in zip(prefix, offset_targets(anchor, prefix)):
                        convert = absolute_action if absolute else native_action
                        action = convert(env.robot, world, planned.reshape(2, 7)[:, 6])
                        if not np.isfinite(action).all():
                            raise ValueError('Non-finite controller action')
                        bounded = np.clip(action, env.action_space.low, env.action_space.high)
                        clipped += int(not np.allclose(bounded, action))
                        actions.append(bounded)
                        obs, _, terminated, truncated, _ = env.step(bounded)
                        success = bool(env.success)
                        video.frame(obs)
                        if success or terminated or truncated or len(actions) >= limit:
                            break
                raw = dict(env._get_task_info())
                steps = len(actions)
                metrics = {k: float(raw[v]) if v in raw else None for k, v in RAW_KEYS.items()}
                metrics['success'] = float(success)
                metrics['tl'] = float(steps)
            finally:
                video.close()
            record = {'task': a.task, 'seed': seed, 'policy_seed': seed,
                      'success': success, 'steps': steps, 'queries': len(queries),
                      'terminated': bool(terminated), 'truncated': bool(truncated),
                      'clipped_steps': clipped, 'metrics': metrics, 'raw_metrics': raw,
                      'simulated_seconds': steps / 20., 'wall_seconds': time.time() - start,
                      'native_completion_time_semantics': 'wall_clock_includes_inference_and_recording',
                      'initial_qpos': initial_qpos, 'initial_scene_qpos': initial_scene_qpos,
                      'initial_scene_qvel': initial_scene_qvel,
                      'initial_body_pos': initial_body_pos, 'initial_body_quat': initial_body_quat,
                      'controller_mode': 'absolute_ee' if absolute else 'delta_ee',
                      'policy_checkpoint_sha256': metadata['checkpoint_sha256'],
                      'query_info': queries, 'video_path': str(video_path),
                      'comparison_status': config['comparison_status']}
            np.savez_compressed(out / f'actions_{seed}.npz', actions=np.asarray(actions))
            write_json(path, record)
            records.append(plain(record))
            write_json(out / 'summary.json', {
                'episodes': len(records), 'successes': sum(r['success'] for r in records),
                'success_rate': float(np.mean([r['success'] for r in records])),
                'comparison_status': config['comparison_status'],
                'metrics': {k: float(np.mean([r['metrics'][k] for r in records]))
                            if all(r['metrics'][k] is not None for r in records) else None for k in metrics},
                'uncertainty': 'Not inferred from small diagnostic samples; official CI estimator pending',
            })
            print(f'{a.task} seed={seed} success={success} steps={steps}', flush=True)
    finally:
        if conn is not None:
            try:
                conn.send({'op': 'close'})
            except (BrokenPipeError, EOFError, ConnectionResetError):
                pass
            conn.close()
        env.close()


if __name__ == '__main__':
    main()
