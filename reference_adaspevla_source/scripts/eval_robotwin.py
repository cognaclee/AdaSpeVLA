"""Run sequential spline smoke evaluation with RoboTwin action dimensions."""

from adaspevla.utils.smoke_evaluation import run_smoke_evaluation


if __name__ == "__main__":
    run_smoke_evaluation("RoboTwin", "beat_block_hammer", allow_dual_arm=True)
