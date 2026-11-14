"""Convert a RoboCasa365 LeRobot v2.1 task dataset to AdaSpeVLA's contract.

Example:
    python -m scripts.benchmarks.convert_robocasa \
        --source_dir datasets/v1.0/target/atomic/CloseBlenderLid/20250822/lerobot \
        --output_dir ./data/converted/robocasa --task_name CloseBlenderLid
"""

import argparse

from adaspevla.benchmarks.robocasa import DEFAULT_CAMERAS, convert_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source_dir", required=True, help="LeRobot dataset root (has meta/info.json)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_name", required=True, help="RoboCasa environment name")
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument(
        "--base_motion", choices=["error", "drop", "ignore"], default="error",
        help="Episodes that drive the mobile base: stop (default), skip, or keep",
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    meta = convert_dataset(
        args.source_dir, args.output_dir, args.task_name, args.cameras,
        args.base_motion, args.max_episodes,
    )
    print(
        f"Converted {meta['episodes']} episodes (dropped {meta['dropped_base_motion_episodes']} "
        f"with base motion); cameras={meta['cameras']} median dt={meta['sample_dt']:g}s"
    )


if __name__ == "__main__":
    main()
