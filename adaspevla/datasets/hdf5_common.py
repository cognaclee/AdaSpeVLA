"""Strict task-scoped HDF5 loading and explicitly requested synthetic test data."""

from pathlib import Path
import h5py
import numpy as np
from .bspline_dataset import UniversalBSplineDataset


def synthetic_episodes(task_name, action_dim=7, chunk_horizon=16):
    """Small deterministic fixtures, only used by an explicit synthetic_data flag."""
    rng = np.random.default_rng(0)
    episodes = []
    for _ in range(2):
        T = max(24, chunk_horizon + 4)
        times = np.arange(T) * 0.02
        actions = np.zeros((T, action_dim), dtype=np.float32)
        for arm in range(action_dim // 7):
            actions[:, 7 * arm] = 0.03 * np.sin(times * 8)
            actions[:, 7 * arm + 6] = np.where(times < times[T // 2], -1.0, 1.0)
        episodes.append(
            dict(
                images=rng.integers(0, 256, (T, 2, 32, 32, 3), dtype=np.uint8),
                actions=actions,
                timestamps=times,
                instruction=f"perform {task_name}",
                action_type="twist",
                action_frame="base",
                gripper_open_value=1.0,
                gripper_closed_value=-1.0,
            )
        )
    return episodes


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def dataset_inventory(group):
    """Report schema only, without dumping images or action values."""
    rows = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            rows.append(f"{name}: shape={obj.shape}, dtype={obj.dtype}")

    group.visititems(visit)
    return "\n".join(rows) or "(no datasets)"


def _array(group, paths, required=True):
    for path in paths:
        if path in group and isinstance(group[path], h5py.Dataset):
            return np.asarray(group[path])
    if required:
        raise ValueError(
            f"Missing required dataset; expected one of {paths} in "
            f"{group.file.filename}:{group.name}.\n"
            f"Available datasets:\n{dataset_inventory(group)}\n"
            "Select camera datasets explicitly with --camera_paths, or a combined "
            "image dataset with --images_path. Use --actions_path for a converted action dataset."
        )
    return None


def load_hdf5_dataset(
    data_dir,
    task_name,
    action_dim,
    chunk_horizon,
    num_control_points,
    degree,
    benchmark,
    synthetic_data=False,
    dataset_options=None,
    camera_paths=None,
    images_path=None,
    actions_path=None,
    **overrides,
):
    if camera_paths is not None:
        if (
            isinstance(camera_paths, str)
            or not camera_paths
            or len(set(camera_paths)) != len(camera_paths)
        ):
            raise ValueError(
                "camera_paths must be a nonempty list of distinct HDF5 dataset paths"
            )
    if camera_paths is not None and images_path is not None:
        raise ValueError("Choose camera_paths or images_path, not both")
    options = dict(dataset_options or {})
    if camera_paths is not None:
        if "num_views" in options and options["num_views"] != len(camera_paths):
            raise ValueError("num_views conflicts with camera_paths")
        options["num_views"] = len(camera_paths)
    allowed = {
        "action_type",
        "action_frame",
        "sample_dt",
        "gripper_open_value",
        "gripper_closed_value",
    }
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown action metadata: {sorted(unknown)}")
    if synthetic_data:
        episodes = synthetic_episodes(task_name, action_dim, chunk_horizon)
    else:
        root_dir = Path(data_dir)
        task_dir = root_dir / task_name
        # Recursion is permitted only inside an explicitly selected task subdirectory.
        if task_dir.is_dir():
            files = sorted(set(task_dir.rglob("*.h5")) | set(task_dir.rglob("*.hdf5")))
        else:
            files = sorted(set(root_dir.glob("*.h5")) | set(root_dir.glob("*.hdf5")))
        if not files:
            raise FileNotFoundError(
                f"No HDF5 demonstrations for {task_name} in {data_dir}"
            )
        episodes = []
        for path in files:
            with h5py.File(path, "r") as root:
                data = root["data"] if "data" in root else root
                keys = sorted(
                    k
                    for k in data
                    if k.startswith("demo") and isinstance(data[k], h5py.Group)
                )
                groups = [data[k] for k in keys] if keys else [data]
                for group in groups:
                    declared_task = _decode(
                        group.attrs.get(
                            "task_name", root.attrs.get("task_name", task_name)
                        )
                    )
                    if declared_task != task_name:
                        raise ValueError(
                            f"{path}: task metadata {declared_task!r} does not match {task_name!r}"
                        )
                    if camera_paths is not None:
                        views = [_array(group, [name]) for name in camera_paths]
                        if any(
                            view.ndim != 4
                            or (view.shape[-1] != 3 and view.shape[1] != 3)
                            for view in views
                        ):
                            raise ValueError(
                                "camera_paths must select decoded RGB arrays (T,H,W,3) or (T,3,H,W); compressed image bytes require decoding before loading"
                            )
                        views = [
                            view if view.shape[-1] == 3 else view.transpose(0, 2, 3, 1)
                            for view in views
                        ]
                        if any(view.shape != views[0].shape for view in views):
                            raise ValueError(
                                "Selected cameras must have matching frame counts and RGB resolution"
                            )
                        images = np.stack(views, axis=1)
                    elif images_path is not None:
                        images = _array(group, [images_path])
                    elif "images" in group:
                        images = _array(group, ["images"])
                    elif benchmark == "robotwin":
                        head = _array(group, ["observation/head_camera/rgb"])
                        wrist = _array(group, ["observation/wrist_camera/rgb"])
                        images = np.stack((head, wrist), axis=1)
                    else:
                        head = _array(
                            group, ["obs/agentview_image", "obs/frontview_image"]
                        )
                        wrist = _array(group, ["obs/robot0_eye_in_hand_image"])
                        images = np.stack((head, wrist), axis=1)
                    actions = _array(
                        group, [actions_path] if actions_path else ["actions", "action"]
                    )
                    instruction = _decode(
                        group.attrs.get(
                            "instruction",
                            root.attrs.get("instruction", f"perform {task_name}"),
                        )
                    )
                    for key in ("instruction", "ep_meta/lang"):
                        if key in group:
                            instruction = _decode(group[key][()])
                            break
                    ep = dict(images=images, actions=actions, instruction=instruction)
                    for key in (
                        "action_type",
                        "action_frame",
                        "sample_dt",
                        "gripper_open_value",
                        "gripper_closed_value",
                    ):
                        recorded = _decode(group.attrs.get(key, root.attrs.get(key)))
                        supplied = overrides.get(key)
                        if (
                            supplied is not None
                            and recorded is not None
                            and supplied != recorded
                        ):
                            raise ValueError(
                                f"{path}: explicit {key} conflicts with HDF5 metadata"
                            )
                        if supplied is not None or recorded is not None:
                            ep[key] = supplied if supplied is not None else recorded
                    for key in (
                        "timestamps",
                        "action_durations",
                        "criticality",
                        "query_intervals",
                        "terminated",
                        "dones",
                    ):
                        value = _array(group, [key], required=False)
                        if value is not None:
                            ep[key] = value
                    if "termination_time" in group.attrs:
                        ep["termination_time"] = float(group.attrs["termination_time"])
                    episodes.append(ep)
    return UniversalBSplineDataset(
        episodes,
        chunk_horizon,
        num_control_points,
        degree,
        action_dim,
        **options,
    )
