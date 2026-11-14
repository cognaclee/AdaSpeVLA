"""Load explicitly converted, timestamped RoboCasa arm demonstrations."""

from .hdf5_common import load_hdf5_dataset


def load_robocasa_hdf5_dataset(
    data_dir,
    task_name="PnPCounterToCab",
    chunk_horizon=16,
    num_control_points=8,
    degree=3,
    action_dim=7,
    synthetic_data=False,
    **action_metadata,
):
    return load_hdf5_dataset(
        data_dir,
        task_name,
        action_dim,
        chunk_horizon,
        num_control_points,
        degree,
        "robocasa",
        synthetic_data,
        **action_metadata,
    )
