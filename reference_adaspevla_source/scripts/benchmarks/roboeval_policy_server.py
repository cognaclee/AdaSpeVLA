"""Local Unix-socket policy process for the isolated RoboEval simulator."""
import argparse,os,hashlib
from multiprocessing.connection import Listener
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from adaspevla.benchmarks.rollout import load_policy,PrefixPlanner
from adaspevla.benchmarks.pose_math import proprio_features

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--socket',required=True);a=p.parse_args()
 torch.set_num_threads(4)
 policy,cal,_=load_policy(a.checkpoint,'cuda');planner=PrefixPlanner(policy,.05)
 metadata={'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
           'model_config':policy.config,'training_stage':'timing'}
 socket=Path(a.socket)
 if socket.exists():raise RuntimeError(f'Refuse to replace existing socket: {socket}')
 listener=Listener(str(socket),family='AF_UNIX');os.chmod(socket,0o600)
 print('READY',flush=True)
 try:
  with listener.accept() as conn:
   while True:
    request=conn.recv()
    if request['op']=='close':break
    if request['op']=='metadata':conn.send(metadata);continue
    if request['op']=='reset':
     planner.reset();torch.manual_seed(request['seed']);torch.cuda.manual_seed_all(request['seed']);conn.send({'ok':True});continue
    try:
     images=torch.from_numpy(np.asarray(request['images'])).permute(0,3,1,2).float()[None].cuda()/255
     proprio=proprio_features(request['poses'],request['gripper'],[Rotation.identity()]*2)
     with torch.inference_mode():actions,info=planner.plan({'images':images,'instructions':[request['instruction']],'proprio':torch.from_numpy(proprio)[None].cuda()})
     assert np.isfinite(actions).all()
     conn.send({'actions':actions,'info':info})
    except Exception as exc:
     conn.send({'error':repr(exc)});raise
 finally:
  listener.close();socket.unlink(missing_ok=True)
if __name__=='__main__':main()
