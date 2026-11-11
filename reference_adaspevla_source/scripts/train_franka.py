"""Train on all selected, verified Franka measured-pose episodes."""
import argparse
import json
from pathlib import Path

from adaspevla.datasets.franka_lerobot import validate_mapping
from adaspevla.datasets.hdf5_common import load_hdf5_dataset
from adaspevla.utils.training import train_two_stage, window_options
from scripts.benchmarks.common_args import add_training_arguments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', type=Path, required=True)
    p.add_argument('--task_name', required=True)
    p.add_argument('--inspect_only', action='store_true', help='Validate and build one sample; do not train')
    add_training_arguments(p, './checkpoints/franka')
    p.set_defaults(control_dt=0.1, min_query_interval=0.1, max_query_interval=0.5,
                   image_fit='pad', vision_finetune_layers=0)
    a = p.parse_args()
    task_dir = a.data_dir / a.task_name
    meta = json.loads((task_dir / 'manifest.json').read_text())
    if meta.get('source') != 'franka_lerobot_v21' or meta.get('complete') is not True or meta.get('task') != a.task_name:
        raise ValueError('Require a complete Franka conversion manifest for this task')
    validate_mapping(meta['mapping'])
    expected = {x['file'] for x in meta['episodes']}
    actual = {str(x.relative_to(task_dir)) for x in task_dir.rglob('*.h5')} | {str(x.relative_to(task_dir)) for x in task_dir.rglob('*.hdf5')}
    if not expected or actual != expected:
        raise ValueError('HDF5 files differ from manifest; refusing to mix conversions')
    if a.max_idle_rows is not None:
        raise ValueError('Franka preserves recorded timing; idle trimming/retiming is disabled')
    options = window_options(a)
    options['num_views'] = 2
    ds = load_hdf5_dataset(a.data_dir, a.task_name, 7, a.chunk_horizon,
                          a.num_control_points, a.degree, 'franka',
                          dataset_options=options, lazy_images=True)
    ds.calibration['source_conversion'] = meta
    print(f'Franka: {len(meta["episodes"])} episodes, {len(ds)} windows, all selected episodes used')
    if a.inspect_only:
        sample = ds[0]
        print({k: list(v.shape) if hasattr(v, 'shape') else type(v).__name__ for k, v in sample.items()})
        return
    train_two_stage(a, ds, 'franka', 7)


if __name__ == '__main__':
    main()
