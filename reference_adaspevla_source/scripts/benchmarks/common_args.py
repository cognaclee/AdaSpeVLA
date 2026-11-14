"""Training arguments shared by the benchmark entry points.

Names and defaults match scripts/train_robotwin.py and scripts/train_robocasa.py
so that ``train_two_stage`` receives the same configuration.
"""

import torch

from adaspevla.utils.training import add_method_arguments


def add_training_arguments(parser, default_output):
    parser.add_argument(
        "--vlm_name",
        default="pi05",
        help="A Hugging Face SigLIP model, or the encoder of pi0, pi05 or gr00t",
    )
    parser.add_argument("--num_control_points", type=int, default=8)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--chunk_horizon", type=int, default=16, help="Action chunk length")
    parser.add_argument("--epochs", type=int, default=10, help="Stage-1 epochs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", default=default_output)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--test_backbone", action="store_true", help="Explicit small offline test backbone"
    )
    parser.add_argument(
        "--horizon_duration",
        type=float,
        default=None,
        help="Inference default in seconds; defaults to median training horizon",
    )
    add_method_arguments(parser)
    return parser
