"""Per-task settings and the recipe knobs. The TASK environment variable picks a task (default "bread");
configs.py, the run scripts, calibrate_timing_anchor.py and the checks all read it, so they always agree.
No openpi imports here.

motion_scale / e_crit come from calibrate_timing_anchor.py on that task's data with this recipe's
ACTION_HORIZON (never reuse another task's or another horizon's: motion_scale is a percentile of offsets
within the window, so it grows with the window length).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root: this file is <root>/adaspe_train/tasks.py

# ---- Recipe (2026-11-10 revision; README original in brackets) ------------------------------------
CONTROL_DT = 0.1
ACTION_HORIZON = 16          # [13] 1.5 s at 10 Hz (frames 0..15); spline time span (H-1)*dt
NUM_CONTROL_POINTS = 11      # [8]  cubic, 8 spans; fit error on these tasks <= the old 13/8 setting
MIN_QUERY_INTERVAL = 0.2     # [0.2]
MAX_QUERY_INTERVAL = round((ACTION_HORIZON - 1) * CONTROL_DT, 6)  # [0.8] = horizon: no forced truncation
LABEL_MODE = "measured_cmd_gripper"  # ["measured"] gripper events from the command, not the lagging width
KNOT_MOTION_NORM = "e_crit"  # ["none"] knot density tanh(motion_variation / e_crit), no saturation

TASKS = {
    "bread": dict(
        repo_id="dddddsds31341/bread-lerobot",  # revision a45b8ba49fc164fb05d8a9c8bdf871ff97004357
        # calibrated 2026-11-10, H=16 (assets/calibration/bread_H16.json): 28,380 windows; labels 0.2s 11.1%, 1.5s 56.3%; near-uniform knots 6.9%
        motion_scale=(0.014774090517312283, 0.07370315981097518, 0.0697999051772058,
                      0.05025116305325356, 0.04219388803843459, 0.04231860391100981),
        e_crit=10.008498916557818,
    ),
    # dataset revision 70bcd1f86ed35fc93a0422a564472355242a51ab; 102 episodes / 16,403 frames.
    "insert_tube": dict(
        repo_id="dddddsds31341/insert-tube-lerobot",
        # calibrated 2026-11-10, H=16 (assets/calibration/insert_tube_H16.json): 16,403 windows; labels 0.2s 12.0%, 1.5s 54.6%; near-uniform knots 4.0%
        motion_scale=(0.007928915799129752, 0.046369556488934885, 0.05284280814230442,
                      0.04649512058816253, 0.032865048675048844, 0.035888442776627796),
        e_crit=11.984896085141408,
    ),
    # dataset revision 2c8cfa8ddef1f83cb09ca7881a0944cd637cd6d0; 112 episodes / 29,795 frames.
    "wipe_whiteboard": dict(
        repo_id="dddddsds31341/wipe-whiteboard-lerobot",
        # calibrated 2026-11-10, H=16 (assets/calibration/wipe_whiteboard_H16.json): 29,795 windows; labels 0.2s 10.2%, 1.5s 60.0%; near-uniform knots 2.8%
        motion_scale=(0.026256428007036446, 0.054885655641555786, 0.040672709792852396,
                      0.03970766179244923, 0.04036529353224039, 0.05067657258583413),
        e_crit=8.95613814977117,
    ),
}

TASK = os.environ.get("TASK", "bread")
if TASK not in TASKS:
    raise ValueError(f"TASK={TASK!r} is not one of {sorted(TASKS)}")
SPEC = TASKS[TASK]


def timing_labels_path(task: str = TASK) -> str:
    """calibrate_timing_anchor.py output that tests/check_timing_labels.py compares against."""
    return f"{ROOT}/adaspe_train/assets/timing_labels/{task}_H{ACTION_HORIZON}.npy"


def calibration_path(task: str = TASK) -> str:
    return f"{ROOT}/adaspe_train/assets/calibration/{task}_H{ACTION_HORIZON}.json"
