"""Franka LeRobot v2.1 ingestion with explicit measured-pose semantics.

Raw commands are preserved, never mistaken for Cartesian pose targets. A mapping
must document its evidence and be verified before producing training data.
"""
import hashlib
import json
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

REVISIONS = {
    'bread': 'a45b8ba49fc164fb05d8a9c8bdf871ff97004357',
    'tennis-balls': '04a27cd5ffe22c250f0a23966256dc646b724a98',
    'insert-tube': '70bcd1f86ed35fc93a0422a564472355242a51ab',
    'test-tube': '29d821a1a71180948773e29c791f906306ac8018',
    'wipe-whiteboard': '2c8cfa8ddef1f83cb09ca7881a0944cd637cd6d0',
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_episode(path, info, tasks):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    required = ('image', 'wrist_image', 'state', 'actions', 'timestamp',
                'frame_index', 'episode_index', 'task_index')
    missing = set(required) - set(table.column_names)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    rows = table.select(required).to_pydict()
    n = len(rows['timestamp'])
    if n < 2:
        raise ValueError('Need at least two samples per episode')
    state = np.asarray(rows['state'], dtype=np.float64)
    actions = np.asarray(rows['actions'], dtype=np.float64)
    times = np.asarray(rows['timestamp'], dtype=np.float64).reshape(-1)
    if state.shape != (n, 19) or actions.shape != (n, 7):
        raise ValueError(f'Expected state (T,19), actions (T,7); got {state.shape}, {actions.shape}')
    if not all(np.isfinite(x).all() for x in (state, actions, times)):
        raise ValueError('Nonfinite state/actions/timestamps')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must strictly increase')
    if not np.allclose(np.diff(times), 1 / float(info['fps']), rtol=1e-3, atol=1e-5):
        raise ValueError('Timestamps disagree with declared fps; do not silently resample')
    if not np.array_equal(np.asarray(rows['frame_index']).reshape(-1), np.arange(n)):
        raise ValueError('Nonconsecutive frame_index; review missing frames or trimming')
    ids = np.unique(rows['episode_index'])
    task_ids = np.unique(rows['task_index'])
    if len(ids) != 1 or len(task_ids) != 1:
        raise ValueError('A file must contain one episode and one task')
    instruction = tasks.get(int(task_ids[0]))
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('No language instruction for task_index')
    return dict(rows=rows, state=state, actions=actions, timestamps=times,
                episode_index=int(ids[0]), instruction=instruction)


def decode_rgb(value, shape):
    if not isinstance(value, dict) or not value.get('bytes'):
        raise ValueError('Expected embedded image bytes in Parquet')
    with Image.open(BytesIO(value['bytes'])) as im:
        frame = np.asarray(im.convert('RGB'))
    if frame.shape != tuple(shape):
        raise ValueError(f'Image shape {frame.shape} != declared {shape}')
    return frame


def audit_episode(ep, info):
    # Decode every image of the selected episode, not just the first thumbnail.
    for key in ('image', 'wrist_image'):
        for value in ep['rows'][key]:
            decode_rgb(value, info['features'][key]['shape'])
    return dict(episode_index=ep['episode_index'], frames=len(ep['state']),
                instruction=ep['instruction'],
                dt_min=float(np.diff(ep['timestamps']).min()),
                dt_max=float(np.diff(ep['timestamps']).max()),
                state_min=ep['state'].min(0).tolist(), state_max=ep['state'].max(0).tolist(),
                action_min=ep['actions'].min(0).tolist(), action_max=ep['actions'].max(0).tolist())


def validate_mapping(mapping):
    if mapping.get('schema_version') != 1:
        raise ValueError('Unsupported mapping schema_version')
    if mapping.get('verified') is not True or not str(mapping.get('evidence', '')).strip():
        raise ValueError('Mapping is unverified: supply acquisition-code evidence; dimension guesses cannot train')
    if mapping.get('supervision') != 'measured_pose':
        raise ValueError('Only measured_pose supervision is implemented; raw actions are not pose targets')
    if mapping.get('output_frame', 'base') not in ('base', 'episode_reset'):
        raise ValueError('output_frame must be base or episode_reset')
    if mapping.get('output_frame') == 'episode_reset' and mapping.get('pose_frame') != 'reset':
        raise ValueError('episode_reset output requires reset-relative source poses')
    if mapping.get('pose_frame') not in ('base', 'reset'):
        raise ValueError('pose_frame must be base or reset')
    if mapping.get('position_unit') != 'm' or mapping.get('angle_unit') != 'rad':
        raise ValueError('Mapping must explicitly produce metres and radians')
    if mapping.get('orientation') not in ('euler_xyz', 'rotvec', 'quaternion_xyzw', 'quaternion_wxyz'):
        raise ValueError('Declare orientation representation explicitly')
    expected = 4 if mapping['orientation'].startswith('quaternion') else 3
    for key, count in (('position_indices', 3), ('orientation_indices', expected)):
        values = mapping.get(key, [])
        if len(values) != count or len(set(values)) != count or any(type(v) is not int or not 0 <= v < 19 for v in values):
            raise ValueError(f'Invalid {key}')
    grip = mapping.get('gripper_index')
    if type(grip) is not int or not 0 <= grip < 19:
        raise ValueError('Invalid gripper_index')
    indices = mapping['position_indices'] + mapping['orientation_indices'] + [grip]
    if len(set(indices)) != len(indices):
        raise ValueError('Pose and gripper channels overlap')
    for key in ('state_scale', 'state_offset'):
        a = np.asarray(mapping.get(key), dtype=float)
        if a.shape != (19,) or not np.isfinite(a).all():
            raise ValueError(f'{key} must explicitly contain 19 finite values')
    if np.any(np.asarray(mapping['state_scale']) == 0):
        raise ValueError('state_scale cannot erase channels')
    opened, closed = mapping.get('gripper_open'), mapping.get('gripper_closed')
    if opened is None or closed is None or not np.isfinite([opened, closed]).all() or opened == closed:
        raise ValueError('Declare distinct finite gripper endpoints')
    tolerance = mapping.get('gripper_clip_tolerance')
    if tolerance is None or not np.isfinite(tolerance) or not 0 <= tolerance <= 0.1:
        raise ValueError('Declare gripper_clip_tolerance in normalized width units, between 0 and 0.1')


def measured_poses(ep, mapping, reset_pose=None):
    validate_mapping(mapping)
    state = ep['state'] * np.asarray(mapping['state_scale']) + np.asarray(mapping['state_offset'])
    if not np.isfinite(state).all():
        raise ValueError('Affine state decoding produced nonfinite values')
    xyz = state[:, mapping['position_indices']]
    orient = state[:, mapping['orientation_indices']]
    kind = mapping['orientation']
    if kind == 'euler_xyz':
        rotation = Rotation.from_euler('xyz', orient)
    elif kind == 'rotvec':
        rotation = Rotation.from_rotvec(orient)
    else:
        quat = orient[:, [1, 2, 3, 0]] if kind == 'quaternion_wxyz' else orient
        if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-3):
            raise ValueError('State quaternion is not unit length')
        rotation = Rotation.from_quat(quat)
    if mapping.get('output_frame') == 'episode_reset':
        if reset_pose is not None:
            raise ValueError('Reset-frame output preserves source coordinates; omit reset transforms')
    elif mapping['pose_frame'] == 'reset':
        # T_base_tcp = T_base_reset @ T_reset_tcp. Never assume identity.
        reset_pose = np.asarray(reset_pose, dtype=float)
        if reset_pose.shape != (7,) or not np.isfinite(reset_pose).all() or not np.isclose(np.linalg.norm(reset_pose[3:]), 1, atol=1e-3):
            raise ValueError('Reset-relative pose requires this episode\'s measured base_from_reset xyz+wxyz')
        base_rotation = Rotation.from_quat(reset_pose[[4, 5, 6, 3]])
        xyz = base_rotation.apply(xyz) + reset_pose[:3]
        rotation = base_rotation * rotation
    elif reset_pose is not None:
        raise ValueError('Do not supply a reset transform for base-frame observations')
    q = rotation.as_quat()[:, [3, 0, 1, 2]]
    raw = state[:, mapping['gripper_index']]
    width = (raw - mapping['gripper_closed']) / (mapping['gripper_open'] - mapping['gripper_closed'])
    tolerance = mapping['gripper_clip_tolerance']
    if np.any(width < -tolerance) or np.any(width > 1 + tolerance):
        raise ValueError('Gripper outside declared endpoints/tolerance; verify units or calibration')
    clipped = int(np.count_nonzero((width < 0) | (width > 1)))
    return np.concatenate((xyz, q), axis=1)[:, None, :], np.clip(width, 0, 1)[:, None], clipped


def write_episode(ep, info, mapping, destination, provenance, reset_pose=None):
    poses, widths, clipped = measured_poses(ep, mapping, reset_pose)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.h5.partial')
    n = len(poses)
    shape = tuple(info['features']['image']['shape'])
    if tuple(info['features']['wrist_image']['shape']) != shape:
        raise ValueError('Camera dimensions differ')
    try:
        with h5py.File(temporary, 'x') as h:
            h.attrs.update(task_name=provenance['task'], instruction=ep['instruction'],
                           action_type='ee_pose', action_frame=mapping.get('output_frame', 'base'),
                           gripper_open_value=1., gripper_closed_value=0.,
                           nominal_sample_dt=1 / float(info['fps']),
                           source_conversion=json.dumps(provenance), mapping=json.dumps(mapping),
                           supervision='measured_pose', gripper_clipped_samples=clipped,
                           time_source='exported nominal frame clock, not verified wall time')
            # Preserve original float32-derived timestamps; no retiming or padding here.
            h.create_dataset('timestamps', data=ep['timestamps'])
            h.create_dataset('ee_poses', data=poses)
            h.create_dataset('gripper', data=widths)
            h.create_dataset('raw/state', data=ep['state'])
            h.create_dataset('raw/actions', data=ep['actions'])
            h.create_dataset('raw/frame_index', data=ep['rows']['frame_index'])
            if reset_pose is not None:
                h.create_dataset('raw/base_from_reset', data=reset_pose)
            image_ds = h.create_dataset('images', (n, 2, *shape), dtype='u1',
                                        chunks=(1, 2, *shape), compression='lzf')
            for i in range(n):
                for view, key in enumerate(('image', 'wrist_image')):
                    image_ds[i, view] = decode_rgb(ep['rows'][key][i], shape)
        temporary.rename(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return dict(file=destination.name, episode_index=ep['episode_index'], frames=n,
                gripper_clipped_samples=clipped, gripper_raw_min=float(ep['state'][:, mapping['gripper_index']].min()),
                gripper_raw_max=float(ep['state'][:, mapping['gripper_index']].max()),
                sha256=digest(destination), **provenance)
