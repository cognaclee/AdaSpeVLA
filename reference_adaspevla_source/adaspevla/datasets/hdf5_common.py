"""Strict task-scoped HDF5 loading and explicitly requested synthetic test data."""

from pathlib import Path
from io import BytesIO
from PIL import Image
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
            actions[:, 7 * arm + 5] = 0.2 * np.cos(times * 6)
            # Continuous width: a smooth opening transition, not a binary step.
            actions[:, 7 * arm + 6] = np.tanh((times - times[T // 2]) / 0.03)
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


def read_rgb_camera(group, path, robotwin_codec=False):
    """Read decoded RGB arrays or individually encoded JPEG/PNG frames."""
    values = _array(group, [path])
    if values.ndim == 4 and (values.shape[-1] == 3 or values.shape[1] == 3):
        return values if values.shape[-1] == 3 else values.transpose(0, 2, 3, 1)
    if values.ndim != 1 or values.dtype.kind not in ("S", "O"):
        raise ValueError(
            f"{path}: expected RGB arrays or a one-dimensional encoded-frame dataset"
        )
    frames = []
    for index in range(len(values)):
        # A slice retains all fixed-width bytes, including terminal NUL bytes.
        blob = (
            values[index : index + 1].tobytes()
            if values.dtype.kind == "S"
            else values[index]
        )
        if isinstance(blob, np.ndarray):
            blob = blob.tobytes()
        if not isinstance(blob, (bytes, bytearray)):
            raise ValueError(f"{path} frame {index}: expected encoded image bytes")
        try:
            with Image.open(BytesIO(blob)) as image:
                frame = np.asarray(image.convert("RGB"))
                if robotwin_codec:
                    # Official codec: unmarked legacy buffers store reversed RGB;
                    # XPL-RGB1 JPEG COM marks standard-order encoded RGB.
                    marked = any(
                        kind == "COM" and payload == b"XPL-RGB1"
                        for kind, payload in getattr(image, "applist", [])
                    )
                    if not marked:
                        frame = frame[..., ::-1].copy()
                frames.append(frame)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{group.file.filename}:{path} frame {index}: could not decode image"
            ) from exc
    if not frames or any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError(
            f"{path}: empty camera stream or inconsistent image resolution"
        )
    return np.stack(frames)


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
    robotwin_native_right=False,
    robotwin_base_wxyz=None,
    robotwin_simulation_dt=1 / 250,
    lazy_images=False,
    **overrides,
):
    if robotwin_native_right:
        if benchmark != "robotwin" or action_dim != 7 or actions_path or images_path:
            raise ValueError(
                "Native right-arm mode requires RoboTwin single-arm input and no action/image path override"
            )
        camera_paths = camera_paths or [
            "vision/cam_head/colors",
            "vision/cam_right_wrist/colors",
        ]
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
                    if lazy_images:
                        if camera_paths is not None or images_path is not None or 'images' not in group:
                            raise ValueError('Lazy image loading requires the converted images dataset')
                        from .lazy_hdf5_images import HDF5ImageSequence
                        images = HDF5ImageSequence(path, group['images'].name)
                    elif camera_paths is not None:
                        views = [
                            read_rgb_camera(
                                group, name, robotwin_codec=robotwin_native_right
                            )
                            for name in camera_paths
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
                    if robotwin_native_right:
                        from .robotwin_native import native_right_episode

                        for key, expected in {
                            "action_type": "ee_pose",
                            "action_frame": "base",
                            "gripper_open_value": 1.0,
                            "gripper_closed_value": 0.0,
                        }.items():
                            if (
                                overrides.get(key) is not None
                                and overrides[key] != expected
                            ):
                                raise ValueError(
                                    f"Native conversion sets {key}={expected}; conflicting override"
                                )
                        ep = native_right_episode(
                            group,
                            images,
                            sample_dt=overrides.get("sample_dt"),
                            base_wxyz=robotwin_base_wxyz,
                            simulation_dt=robotwin_simulation_dt,
                        )
                        for key in (
                            "criticality",
                            "query_intervals",
                            "terminated",
                            "dones",
                        ):
                            if key in group:
                                ep[key] = np.asarray(group[key])
                        if "termination_time" in group.attrs:
                            ep["termination_time"] = float(
                                group.attrs["termination_time"]
                            )
                        episodes.append(ep)
                        continue
                    if (
                        actions_path is None
                        and "ee_poses" not in group
                        and "action/left_ee_poses" in group
                        and "action/right_ee_poses" in group
                    ):
                        raise ValueError(
                            f"{path}: this is a raw XPolicyLab export with per-arm state/action poses. "
                            "Camera decoding is supported, but building the pose contract requires the "
                            "quaternion order, reference frame, action/state time alignment and gripper "
                            "coding. Convert it with scripts.benchmarks.convert_xpolicylab "
                            "(or --robotwin_native_right for the right arm)."
                        )
                    if actions_path is None and "ee_poses" in group:
                        # Converted pose contract: (T,arms,7) poses and raw widths.
                        ep_actions = dict(
                            ee_poses=_array(group, ["ee_poses"]),
                            gripper=_array(group, ["gripper"]),
                        )
                        for state_key in ("observed_ee_poses", "observed_gripper"):
                            if state_key in group:
                                ep_actions[state_key] = _array(group, [state_key])
                        base = group.attrs.get("base_wxyz", root.attrs.get("base_wxyz"))
                        if base is not None:
                            ep_actions["base_wxyz"] = np.asarray(base, dtype=np.float64)
                    else:
                        ep_actions = dict(
                            actions=_array(
                                group,
                                [actions_path] if actions_path else ["actions", "action"],
                            )
                        )
                    instruction = _decode(
                        group.attrs.get(
                            "instruction",
                            root.attrs.get("instruction", f"perform {task_name}"),
                        )
                    )
                    for key in ("instruction", "instructions", "ep_meta/lang"):
                        if key in group:
                            instruction = _decode(group[key][()])
                            break
                    ep = dict(images=images, instruction=instruction, **ep_actions)
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
                    frequency = _array(
                        group, ["additional_info/frequency"], required=False
                    )
                    if frequency is not None:
                        if (
                            frequency.ndim != 0
                            or not np.isfinite(frequency)
                            or frequency <= 0
                        ):
                            raise ValueError(
                                "additional_info/frequency must be a positive scalar Hz value"
                            )
                        period = 1.0 / float(frequency)
                        if "sample_dt" in ep and not np.isclose(
                            ep["sample_dt"], period
                        ):
                            raise ValueError(
                                "sample_dt conflicts with additional_info/frequency"
                            )
                        ep["sample_dt"] = period
                    for key in (
                        "timestamps",
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
