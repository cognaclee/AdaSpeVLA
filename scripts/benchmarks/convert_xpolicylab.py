"""Convert XPolicyLab v1.0 trajectories (RoboTwin 2.0 or RoboDojo) to the contract.

Example (RoboTwin, both arms):
    python -m scripts.benchmarks.convert_xpolicylab \
        --source_dir ../data/RoboTwin/beat_block_hammer/aloha_agilex/demo_clean \
        --output_dir ./data/converted/robotwin --task_name beat_block_hammer \
        --profile robotwin --arms left right
"""

import argparse

from adaspevla.benchmarks.xpolicylab_data import convert_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source_dir", required=True, help="Directory of episode HDF5 files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--profile", choices=["robotwin", "robodojo"], default="robotwin")
    parser.add_argument(
        "--arms", nargs="+", default=["right"], choices=["left", "right", "single"],
        help="Arms supplying action targets, in channel order",
    )
    parser.add_argument("--cameras", nargs="+", default=None, help="XPolicyLab camera names")
    parser.add_argument(
        "--base_wxyz", nargs="+", type=float, default=None,
        help="World-from-base quaternion(s): 4 values, or 4 per arm",
    )
    parser.add_argument("--sample_dt", type=float, default=None, help="Override sample period")
    parser.add_argument("--simulation_dt", type=float, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    base = None
    if args.base_wxyz is not None:
        if len(args.base_wxyz) % 4:
            parser.error("--base_wxyz takes 4 values per quaternion")
        base = [args.base_wxyz[i : i + 4] for i in range(0, len(args.base_wxyz), 4)]
        base = base[0] if len(base) == 1 else base
    meta = convert_directory(
        args.source_dir,
        args.output_dir,
        args.task_name,
        max_episodes=args.max_episodes,
        arms=args.arms,
        cameras=args.cameras,
        profile=args.profile,
        base_wxyz=base,
        sample_dt=args.sample_dt,
        simulation_dt=args.simulation_dt,
    )
    print(
        f"Converted {meta['episodes']} episodes: arms={meta['arms']} cameras={meta['cameras']} "
        f"sample_dt={meta['sample_dt']:g}s ({meta['time_source']})"
    )


if __name__ == "__main__":
    main()
