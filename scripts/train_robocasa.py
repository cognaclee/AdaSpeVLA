import argparse
import torch
from adaspevla.utils.training import (
    add_method_arguments,
    window_options,
    train_two_stage,
)

from adaspevla.datasets.robocasa_adapter import load_robocasa_hdf5_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AdaSpeVLA Policy on RoboCasa Kitchen Benchmark"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/robocasa",
        help="RoboCasa HDF5 dataset directory",
    )
    parser.add_argument(
        "--task_name", type=str, default="PnPCounterToCab", help="RoboCasa task name"
    )
    parser.add_argument(
        "--vlm_name",
        type=str,
        default="google/siglip-base-patch16-224",
        help="VLM backbone",
    )
    parser.add_argument(
        "--num_control_points",
        type=int,
        default=8,
        help="Number of B-spline control points",
    )
    parser.add_argument(
        "--degree", type=int, default=3, help="B-spline polynomial degree"
    )
    parser.add_argument(
        "--chunk_horizon", type=int, default=16, help="Action chunk length"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/adaspevla_robocasa",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--synthetic_data",
        action="store_true",
        help="Explicit synthetic fixtures for smoke tests",
    )
    parser.add_argument(
        "--test_backbone",
        action="store_true",
        help="Explicit small offline test backbone",
    )
    parser.add_argument("--action_type", choices=["twist", "delta_twist"], default=None)
    parser.add_argument("--action_frame", choices=["base"], default=None)
    parser.add_argument(
        "--sample_dt",
        type=float,
        default=None,
        help="Demonstration sample period in seconds",
    )
    parser.add_argument("--gripper_open_value", type=float, default=None)
    parser.add_argument("--gripper_closed_value", type=float, default=None)
    parser.add_argument(
        "--horizon_duration",
        type=float,
        default=None,
        help="Inference default in seconds; defaults to median training horizon",
    )
    cameras = parser.add_mutually_exclusive_group()
    cameras.add_argument(
        "--camera_paths",
        nargs="+",
        help="Ordered RGB dataset paths relative to each episode group",
    )
    cameras.add_argument(
        "--images_path",
        help="Combined image dataset path relative to each episode group",
    )
    parser.add_argument(
        "--actions_path",
        help="Converted physical action dataset path; does not convert joint/pose commands",
    )
    add_method_arguments(parser)
    return parser.parse_args()


def train():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    action_dim = 7  # RoboCasa standard single-arm end-effector action

    print("=" * 80)
    print(f" AdaSpeVLA: Training on RoboCasa ({args.task_name.upper()}) ")
    print(
        f" Action Dim: {action_dim} | Control Points: {args.num_control_points} | Degree: {args.degree}"
    )
    print(
        f" VLM Backbone: {'offline test' if args.test_backbone else args.vlm_name} | Device: {args.device}"
    )
    print("=" * 80)

    dataset = load_robocasa_hdf5_dataset(
        data_dir=args.data_dir,
        task_name=args.task_name,
        chunk_horizon=args.chunk_horizon,
        num_control_points=args.num_control_points,
        degree=args.degree,
        action_dim=action_dim,
        synthetic_data=args.synthetic_data,
        dataset_options=window_options(args),
        camera_paths=args.camera_paths,
        images_path=args.images_path,
        actions_path=args.actions_path,
        action_type=args.action_type,
        action_frame=args.action_frame,
        sample_dt=args.sample_dt,
        gripper_open_value=args.gripper_open_value,
        gripper_closed_value=args.gripper_closed_value,
    )
    train_two_stage(args, dataset, "robocasa", action_dim)


if __name__ == "__main__":
    train()
