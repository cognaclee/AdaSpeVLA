"""Regular RoboEval base-task evaluation with paired seeds and native metrics.

Simulator environment remains unmodified. A Unix socket supplies adaptive
anchor-relative prefixes from the independent training/policy environment.
"""
import argparse,json,time
from pathlib import Path
from multiprocessing.connection import Client
import numpy as np
from adaspevla.benchmarks.roboeval_control import make_env,CAMERAS,INSTRUCTIONS,observed_poses,offset_targets,native_action
from roboeval.challenge.results import to_jsonable,episode_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--task',required=True);p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=100);p.add_argument('--seed',type=int,default=1000000);p.add_argument('--max_steps',type=int,default=None);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 limit=a.max_steps or {'PickSingleBookFromTable':300,'LiftTray':200}[a.task]
 env=make_env(a.task);conn=Client(a.socket,family='AF_UNIX');records=[]
 try:
  for seed in range(a.seed,a.seed+a.episodes):
   target=out/f'episode_{seed}.json'
   if target.exists():records.append(json.loads(target.read_text()));continue
   start=time.time();obs,_=env.reset(seed=seed);conn.send({'op':'reset','seed':seed});assert conn.recv()['ok']
   steps=queries=clipped=0;success=False;terminated=truncated=False;actions_log=[];query_info=[]
   while steps<limit and not (success or terminated or truncated):
    anchor=observed_poses(env.robot)
    conn.send({'op':'predict','images':np.stack([obs['rgb_'+c].transpose(1,2,0) for c in CAMERAS]),'poses':anchor,'gripper':env.robot.qpos_grippers,'instruction':INSTRUCTIONS[a.task]})
    response=conn.recv()
    if 'error' in response:raise RuntimeError(response['error'])
    prefix=response['actions'];targets=offset_targets(anchor,prefix);queries+=1;query_info.append(response['info'])
    for planned,world in zip(prefix,targets):
     action=native_action(env.robot,world,planned.reshape(2,7)[:,6]);bounded=np.clip(action,env.action_space.low,env.action_space.high)
     clipped+=int(not np.allclose(bounded,action));actions_log.append(bounded)
     obs,_,terminated,truncated,info=env.step(bounded);steps+=1;success=bool(env.success)
     if success or terminated or truncated or steps>=limit:break
   raw=env._metric_finalize(success_flag=success)
   metrics=episode_metrics(to_jsonable(raw),steps,20);metrics['success']=float(success)
   record={'task':a.task,'seed':seed,'policy_seed':seed,'success':success,'steps':steps,'queries':queries,'wall_seconds':time.time()-start,'clipped_steps':clipped,'terminated':bool(terminated),'truncated':bool(truncated),'metrics':metrics,'raw_metrics':to_jsonable(raw),'query_info':query_info}
   np.savez_compressed(out/f'actions_{seed}.npz',actions=np.asarray(actions_log))
   tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(target)
   records.append(record);print(f'{a.task} seed={seed} success={success} steps={steps} queries={queries}',flush=True)
   summary={'episodes':len(records),'successes':sum(r['success'] for r in records),'success_rate':float(np.mean([r['success'] for r in records])),'metrics':{k:float(np.mean([r['metrics'][k] for r in records])) for k in metrics}}
   (out/'summary.json').write_text(json.dumps(summary,indent=2))
 finally:
  try:conn.send({'op':'close'})
  except (BrokenPipeError,EOFError,ConnectionResetError):pass
  conn.close();env.close()
if __name__=='__main__':main()
