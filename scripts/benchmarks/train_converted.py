"""Train AdaSpeVLA (both stages) on converted RoboTwin 2.0, RoboDojo or RoboCasa365 data.

The conversion metadata (arms, cameras, base frames, sample period, controller
scales) is stored in the checkpoint calibration so the evaluators can deploy the
checkpoint without restating it.
"""

import argparse

from adaspevla.benchmarks.xpolicylab_data import read_conversion
from adaspevla.datasets.robocasa_adapter import load_robocasa_hdf5_dataset
from adaspevla.datasets.robotwin_adapter import load_robotwin_hdf5_dataset
from adaspevla.utils.training import train_two_stage, window_options
from scripts.benchmarks.common_args import add_training_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data_dir", required=True, help="Converter output_dir")
    parser.add_argument("--task_name", required=True)
    add_training_arguments(parser, "./checkpoints/adaspevla_benchmark")
    args = parser.parse_args()
    meta = read_conversion(args.data_dir, args.task_name)
    num_arms = len(meta["arms"])
    options = window_options(args)
    options["num_views"] = len(meta["cameras"])
    common = dict(
        data_dir=args.data_dir,
        task_name=args.task_name,
        chunk_horizon=args.chunk_horizon,
        num_control_points=args.num_control_points,
        degree=args.degree,
        dataset_options=options,
    )
    if meta["source"] == "robocasa365_lerobot_v21":
        benchmark = "robocasa"
        dataset = load_robocasa_hdf5_dataset(**common)
    else:
        benchmark = meta["profile"]
        dataset = load_robotwin_hdf5_dataset(dual_arm=num_arms == 2, **common)
    dataset.calibration["source_conversion"] = meta
    print(
        f"{benchmark} {args.task_name}: arms={meta['arms']} cameras={meta['cameras']} "
        f"sample_dt={meta['sample_dt']:g}s ({meta['time_source']})"
    )
    train_two_stage(args, dataset, benchmark, 7 * num_arms)


if __name__ == "__main__":
    main()
