"""Convert selected RoboEval LeRobot image shards to measured-pose HDF5.

Uses official kinematics in a separate simulator environment. No task-name
fallback and no assumption that native 14-D actions are interleaved arm actions.
"""
import argparse,json,subprocess,sys,os
from pathlib import Path
from io import BytesIO
import numpy as np
import pyarrow.parquet as pq
import h5py
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--task',required=True);p.add_argument('--pose_helper',required=True);p.add_argument('--sim_python',required=True)
    a=p.parse_args();source=Path(a.source)/a.task;out=Path(a.output)/a.task;out.mkdir(parents=True,exist_ok=True)
    cams=['cam_head','cam_left_wrist','cam_right_wrist']
    tasks={r['task_index']:r['__index_level_0__'] for r in pq.read_table(source/'meta/tasks.parquet').to_pylist()}
    episodes={}
    cols=['episode_index','frame_index','timestamp','state','state_gripper','actions','uuid','task_index']+cams
    for f in sorted((source/'data').rglob('*.parquet')):
        for row in pq.read_table(f,columns=cols).to_pylist():episodes.setdefault(int(row['episode_index']),[]).append(row)
    ordered=[];manifest=[]
    for ep,rows in sorted(episodes.items()):
        rows.sort(key=lambda r:r['frame_index']);assert [r['frame_index'] for r in rows]==list(range(len(rows)))
        ordered.extend(rows);manifest.append({'episode_index':ep,'uuid':rows[0]['uuid'],'frames':len(rows)})
    np.savez(out/'states.npz',states=np.asarray([r['state'] for r in ordered],dtype=np.float32))
    subprocess.run([a.sim_python,a.pose_helper,a.task,str(out/'states.npz'),str(out/'poses.npz')],check=True)
    poses=np.load(out/'poses.npz')['poses'];offset=0
    for ep,rows in sorted(episodes.items()):
        T=len(rows); path=out/f'episode_{ep:04d}.h5'
        with h5py.File(str(path)+'.tmp','w') as h:
            h.attrs.update(task_name=a.task,instruction=tasks[rows[0]['task_index']],action_type='ee_pose',action_frame='base',sample_dt=.05,gripper_open_value=1.,gripper_closed_value=0.,uuid=str(rows[0]['uuid']),pose_semantics='measured wrist poses; fixed world-aligned reference',source='roboeval_ee_delta_20hz_scaled')
            h.create_dataset('ee_poses',data=poses[offset:offset+T]);h.create_dataset('gripper',data=np.asarray([r['state_gripper'] for r in rows]))
            h.create_dataset('timestamps',data=np.arange(T,dtype=np.float64)*.05)
            h.create_dataset('source_actions',data=np.asarray([r['actions'] for r in rows]));h.create_dataset('source_state',data=np.asarray([r['state'] for r in rows]))
            imgs=h.create_dataset('images',(T,3,256,256,3),dtype='u1',chunks=(1,3,256,256,3),compression='lzf')
            for t,row in enumerate(rows):
                for v,c in enumerate(cams):
                    with Image.open(BytesIO(row[c]['bytes'])) as im:imgs[t,v]=np.asarray(im.convert('RGB'))
        os.replace(str(path)+'.tmp',path);offset+=T
        print(f'{a.task} episode {ep}: {T} frames',flush=True)
    (out/'manifest.json').write_text(json.dumps({'task':a.task,'cameras':cams,'sample_dt':.05,'episodes':manifest,'frames':offset,'pose_source':'official FK of measured qpos','state_source':'measured EE pose and measured normalized gripper opening','raw_state_dim':36,'policy_proprio_dim':20},indent=2))
if __name__=='__main__':main()
