"""RoboEval bridge: world wrist targets -> native control-reference EE deltas.

Pure numpy/scipy module shared with the isolated simulator environment.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
CAMERAS=('head','left_wrist','right_wrist')
_FAMILY_VARIANTS = {
    'LiftPot': ('LiftPot','LiftPotPosition','LiftPotOrientation','LiftPotPositionAndOrientation'),
    'StackSingleBookShelf': ('StackSingleBookShelf','StackSingleBookShelfPosition','StackSingleBookShelfPositionAndOrientation'),
    'PickSingleBookFromTable': ('PickSingleBookFromTable','PickSingleBookFromTablePosition','PickSingleBookFromTableOrientation','PickSingleBookFromTablePositionAndOrientation'),
    'StackTwoBlocks': ('StackTwoBlocks','StackTwoBlocksPosition','StackTwoBlocksOrientation','StackTwoBlocksPositionAndOrientation'),
    'CubeHandover': ('CubeHandover','CubeHandoverPosition','CubeHandoverOrientation','CubeHandoverPositionAndOrientation'),
    'RotateValve': ('RotateValve','RotateValvePosition','RotateValvePositionAndOrientation'),
    'PackBox': ('PackBox','PackBoxPosition','PackBoxOrientation','PackBoxPositionAndOrientation'),
    'LiftTray': ('LiftTray','LiftTrayPosition','LiftTrayOrientation','LiftTrayPositionAndOrientation'),
}
TASK_FAMILIES = {variant: family for family, variants in _FAMILY_VARIANTS.items() for variant in variants}
INSTRUCTIONS = {
    task: {
        'LiftPot':'Lift the pot off the counter using both hands.',
        'StackSingleBookShelf':'Place the book on the shelf.',
        'PickSingleBookFromTable':'Pick up the book from the table.',
        'StackTwoBlocks':'Stack one block on top of the other.',
        'CubeHandover':'Hand the object from one gripper to the other.',
        'RotateValve':'Rotate both valves counterclockwise.',
        'PackBox':'Close the flaps of the box.',
        'LiftTray':'Lift the tray off the table using both hands.',
    }[family]
    for task, family in TASK_FAMILIES.items()
}

def make_env(task,images=True,*,absolute=False):
    from importlib import import_module
    from roboeval.action_modes import JointPositionActionMode
    from roboeval.robots.configs.panda import BimanualPanda
    from roboeval.utils.observation_config import ObservationConfig,CameraConfig
    family=TASK_FAMILIES[task]
    module={
        'LiftPot':'lift_pot','StackSingleBookShelf':'stack_books',
        'PickSingleBookFromTable':'stack_books','StackTwoBlocks':'manipulation',
        'CubeHandover':'manipulation','RotateValve':'rotate_utility_objects',
        'PackBox':'pack_objects','LiftTray':'lift_tray',
    }[family]
    cls=getattr(import_module(f'roboeval.envs.{module}'),task)
    env=cls(action_mode=JointPositionActionMode(floating_base=True,floating_dofs=[],absolute=absolute,ee=True),robot_cls=BimanualPanda,control_frequency=20,observation_config=ObservationConfig(cameras=[CameraConfig(c,resolution=(256,256)) for c in CAMERAS] if images else [],proprioception=True,privileged_information=False),render_mode=None)
    # Public task/embodiment calibration (the environment loads robot_info
    # overrides), not measured scene or actuator state.
    env.robot._adaspevla_fixed_base=(np.asarray(env.RESET_ROBOT_POS).copy(),
                                    np.asarray(env.RESET_ROBOT_QUAT).copy())
    return env

def proprio_poses(robot,obs):
    """Measured fixed-base Panda wrist poses from declared joint observations."""
    if len(obs.get('proprioception_floating_base', [])):
        raise ValueError('Observable FK adapter currently supports fixed-base Panda only')
    q=np.asarray(obs['proprioception'])
    fk=robot.forward_kinematics(np.r_[q[:7],q[9:16]]).reshape(2,6)
    base_pos,base_q=robot._adaspevla_fixed_base
    rot=R.from_rotvec(fk[:,3:])*R.from_quat(base_q[[1,2,3,0]])
    return np.c_[fk[:,:3]+base_pos,rot.as_quat()[:,[3,0,1,2]]]

def absolute_action(robot,targets,widths):
    """Absolute FK-frame EE targets; no actuator/control reference is read."""
    poses=world_to_fk_frame(robot,np.asarray(targets))
    return np.r_[poses.reshape(-1),1-np.clip(widths,0,1)].astype(np.float32)

def observed_poses(robot):
    return np.asarray([np.r_[g.wrist_position,g.wrist_orientation.elements] for g in robot.grippers.values()])

def reset_observation(env,seed):
    """Refresh derived render state after official reset, without physics steps."""
    env.reset(seed=seed)
    env._mojo.physics.forward()
    return env.get_observation(),env.get_info()

def offset_targets(anchor,actions):
    actions=np.asarray(actions).reshape(-1,2,7)
    poses=np.empty((len(actions),2,7))
    for arm in range(2):
        poses[:,arm,:3]=anchor[arm,:3]+actions[:,arm,:3]
        rotation=R.from_rotvec(actions[:,arm,3:6])*R.from_quat(anchor[arm,[4,5,6,3]])
        poses[:,arm,3:]=rotation.as_quat()[:,[3,0,1,2]]
    return poses

def world_to_fk_frame(robot,targets):
    values=[]
    for target in targets:
        pos=target[:3].copy();rot=R.from_quat(target[[4,5,6,3]])
        if hasattr(robot,'_adaspevla_fixed_base'):
            base_pos,q=robot._adaspevla_fixed_base
            pos-=base_pos;rot=rot*R.from_quat(q[[1,2,3,0]]).inv()
        elif robot._pelvis is not None:
            pos-=robot._pelvis.get_position()
            q=robot._pelvis.get_quaternion();rot=rot*R.from_quat(np.asarray(q)[[1,2,3,0]]).inv()
        values.append(np.r_[pos,rot.as_rotvec()])
    return np.asarray(values)

def native_action(robot,targets,widths):
    reference=robot.forward_kinematics(robot.get_initial_joint_positions()).reshape(2,6)
    targets=world_to_fk_frame(robot,np.asarray(targets))
    delta=np.empty((2,6));delta[:,:3]=targets[:,:3]-reference[:,:3]
    delta[:,3:]=(R.from_rotvec(targets[:,3:])*R.from_rotvec(reference[:,3:]).inv()).as_rotvec()
    # Native array is left6,right6,left_grip,right_grip; Panda reverses controls.
    return np.r_[delta.reshape(-1),1-np.clip(widths,0,1)].astype(np.float32)
