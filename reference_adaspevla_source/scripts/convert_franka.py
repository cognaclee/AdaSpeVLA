"""Audit or convert Franka episodes. No automatic train/validation split."""
import argparse
import json
from pathlib import Path

from adaspevla.datasets.franka_lerobot import (
    REVISIONS, audit_episode, digest, read_episode, validate_mapping, write_episode,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=REVISIONS, required=True)
    p.add_argument('--source_dir', type=Path, help='Local LeRobot repository; otherwise use HF cache/download')
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--mode', choices=['audit', 'convert'], default='audit')
    p.add_argument('--mapping', type=Path)
    p.add_argument('--reset_poses', type=Path, help='JSON: episode index -> measured base_from_reset xyz+wxyz')
    p.add_argument('--max_episodes', type=int, default=None)
    p.add_argument('--episode_indices', type=int, nargs='+')
    a = p.parse_args()
    if a.max_episodes is not None and a.max_episodes < 1:
        p.error('--max_episodes must be positive')
    mapping = None
    if a.mode == 'convert':
        if a.mapping is None:
            p.error('--mapping is required to convert')
        mapping = json.loads(a.mapping.read_text())
        validate_mapping(mapping)
    resets = json.loads(a.reset_poses.read_text()) if a.reset_poses else {}
    repo = f'dddddsds31341/{a.task}-lerobot'
    revision = REVISIONS[a.task]

    def source(relative):
        if a.source_dir:
            return a.source_dir / relative
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=repo, repo_type='dataset', revision=revision, filename=relative))

    info_path, tasks_path = source('meta/info.json'), source('meta/tasks.jsonl')
    info = json.loads(info_path.read_text())
    if info.get('codebase_version') != 'v2.1' or info.get('fps', 0) <= 0:
        raise ValueError('Expected explicit LeRobot v2.1 metadata and positive fps')
    tasks = {x['task_index']: x['task'] for x in map(json.loads, tasks_path.read_text().splitlines())}
    indices = a.episode_indices if a.episode_indices is not None else list(range(info['total_episodes']))
    if len(set(indices)) != len(indices) or any(i < 0 or i >= info['total_episodes'] for i in indices):
        raise ValueError('Invalid or duplicate episode indices')
    if a.max_episodes:
        indices = indices[:a.max_episodes]
    dest = a.output_dir / a.task
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / ('audit.json' if a.mode == 'audit' else 'manifest.json')
    if manifest_path.exists():
        raise FileExistsError(f'{manifest_path}: choose a fresh output directory')
    if a.mode == 'convert' and any(dest.glob('*.h5')):
        raise FileExistsError('Output already has HDF5 files; choose a fresh conversion directory')
    manifest = dict(schema_version=1, source='franka_lerobot_v21', task=a.task,
                    repo_id=repo, revision=revision if not a.source_dir else None,
                    local_source=str(a.source_dir.resolve()) if a.source_dir else None,
                    source_info_sha256=digest(info_path), source_tasks_sha256=digest(tasks_path),
                    nominal_fps=info['fps'], cameras=['image', 'wrist_image'], arms=['franka'],
                    split='all_selected_episodes', mapping=mapping, complete=False, episodes=[])
    for index in indices:
        rel = info['data_path'].format(episode_chunk=index // info['chunks_size'], episode_index=index)
        path = source(rel)
        ep = read_episode(path, info, tasks)
        if ep['episode_index'] != index:
            raise ValueError('File name and episode_index disagree')
        provenance = dict(task=a.task, source_file=rel, source_sha256=digest(path),
                          repo_id=repo, revision=manifest['revision'])
        if a.mode == 'audit':
            row = dict(**audit_episode(ep, info), **provenance)
        else:
            row = write_episode(ep, info, mapping, dest / f'episode_{index:06d}.h5',
                                provenance, resets.get(str(index)))
        manifest['episodes'].append(row)
        print(f'{a.mode}: {a.task}/{index}: {row["frames"]} frames', flush=True)
    manifest['complete'] = True
    manifest['total_frames'] = sum(x['frames'] for x in manifest['episodes'])
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(manifest_path)


if __name__ == '__main__':
    main()
