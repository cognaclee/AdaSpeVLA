"""Run sequential spline smoke evaluation with RoboCasa action dimensions."""

from adaspevla.utils.smoke_evaluation import run_smoke_evaluation


if __name__ == "__main__":
    run_smoke_evaluation("RoboCasa", "PnPCounterToCab")
