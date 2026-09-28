"""Load explicitly converted, timestamped RoboTwin arm demonstrations."""

from .hdf5_common import load_hdf5_dataset


def load_robotwin_hdf5_dataset(
    data_dir,
    task_name="beat_block_hammer",
    dual_arm=False,
    chunk_horizon=16,
    num_control_points=8,
    degree=3,
    synthetic_data=False,
    **action_metadata,
):
    return load_hdf5_dataset(
        data_dir,
        task_name,
        14 if dual_arm else 7,
        chunk_horizon,
        num_control_points,
        degree,
        "robotwin",
        synthetic_data,
        **action_metadata,
    )
