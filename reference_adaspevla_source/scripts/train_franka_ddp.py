"""Resumable AdaSpeVLA Franka DDP training, with atomic rank-0 checkpoints."""
import argparse, hashlib, json, math, os, random, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from adaspevla.models.adaspe_vla_policy import AdaSpeVLAPolicy
from adaspevla.datasets.hdf5_common import load_hdf5_dataset
from adaspevla.datasets.bspline_dataset import collate_windows
from adaspevla.datasets.franka_training import load_franka_training, policy_config

class LossModule(torch.nn.Module):
 def __init__(self,policy):super().__init__();self.policy=policy
 def forward(self,batch):return self.policy.forward_loss(batch)[0]

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--recipe',default=str(Path(__file__).resolve().parents[1]/'configs/franka/recipe_v1.json'));p.add_argument('--test-backbone',action='store_true');p.add_argument('--cpu',action='store_true');p.add_argument('--output',required=True);p.add_argument('--action-steps',type=int,default=None);p.add_argument('--timing-steps',type=int,default=None)
 p.add_argument('--local-batch',type=int,default=None);p.add_argument('--seed',type=int,default=None);p.add_argument('--save-every',type=int,default=None);p.add_argument('--log-every',type=int,default=None)
 p.add_argument('--preparation-report');p.add_argument('--max-updates',type=int);p.add_argument('--resume',action='store_true')
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 assert not a.cpu or a.test_backbone, '--cpu requires --test-backbone'
 recipe=json.loads(Path(a.recipe).read_text());recipe_hash=hashlib.sha256(Path(a.recipe).read_bytes()).hexdigest()
 for key in ['action_steps','timing_steps','local_batch','seed','save_every','log_every']:
  if getattr(a,key) is None:setattr(a,key,recipe[key])
 for key in ['action_steps','timing_steps','local_batch','save_every','log_every']:
  if getattr(a,key)<1:raise ValueError(f'{key} must be positive')
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
 if not a.test_backbone and world != recipe['world_size']:raise ValueError('World size differs from recipe; explicitly revise the recipe for a different GPU allocation')
 if not a.cpu:torch.cuda.set_device(local)
 device=torch.device('cpu') if a.cpu else torch.device('cuda',local);torch.set_num_threads(2 if a.cpu else 4)
 if world>1:dist.init_process_group('gloo' if a.cpu else 'nccl',**({} if a.cpu else dict(device_id=device)))
 def broadcast(v):
  if world>1:dist.broadcast(v,src=0)
 def reduce(v,op=dist.ReduceOp.SUM):
  if world>1:dist.all_reduce(v,op=op)
 def barrier():
  if world>1:dist.barrier()
 def finish():
  if world>1:dist.destroy_process_group()
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 seed_all(a.seed)
 def write_status(**extra):
  if rank==0:
   value=dict(updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),world_size=world,local_batch=a.local_batch,global_batch=a.local_batch*world,**extra)
   tmp=out/'status.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(out/'status.json')
 write_status(status='loading_data')
 joint,data_hash=load_franka_training(a.data_dir,recipe['task_names'],recipe)
 if a.preparation_report:
  prepared=json.loads(Path(a.preparation_report).read_text())
  assert prepared['status']=='passed' and prepared['data_manifest_sha256']==data_hash
  assert prepared['recipe_sha256']==recipe_hash
 elif not a.test_backbone:raise ValueError('Formal training requires --preparation-report from prepare_franka_training')
 config=policy_config(joint,recipe,a.test_backbone)
 policy=AdaSpeVLAPolicy(**config).to(device)
 resume=None
 if a.resume:
  resume=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
  assert resume['world_size']==world and resume['data_manifest_sha256']==data_hash
  assert resume['recipe_sha256']==recipe_hash and resume['model_config']==policy.config
  for key in ['action_steps','timing_steps','local_batch','seed']:
   assert resume['args'][key]==vars(a)[key],key
  policy.load_checkpoint_state(resume['model_state'])
 elif (out/'resume.pt').exists():raise RuntimeError('Existing checkpoint requires --resume')
 def move(v):
  if isinstance(v,torch.Tensor):return v.to(device)
  if isinstance(v,list):return [move(x) for x in v]
  return v
 updates=0
 for si,(stage,total) in enumerate([('action',a.action_steps),('timing',a.timing_steps)]):
  if resume and si<resume['stage_index']:continue
  step=resume['step'] if resume and si==resume['stage_index'] else 0
  if step>=total:continue
  joint.include_diagnostic_targets=stage!='timing';policy.set_training_stage(stage);policy.train()
  wrapper=LossModule(policy)
  wrapper._ddp_params_and_buffers_to_ignore={n for n,v in wrapper.named_parameters() if not v.requires_grad}
  params=[v for v in policy.parameters() if v.requires_grad]
  assert all(not v.requires_grad for n,v in policy.named_parameters() if n.startswith(('vlm.vision_encoder.','vlm.text_encoder.')))
  if stage=='timing':assert all(n.startswith('freq_head.') for n,v in policy.named_parameters() if v.requires_grad)
  for v in params:
   ref=v.detach().clone();broadcast(ref);assert torch.equal(v,ref)
  ddp=DDP(wrapper,device_ids=None if a.cpu else [local],broadcast_buffers=False,find_unused_parameters=True,init_sync=False) if world>1 else wrapper
  optimizer=torch.optim.AdamW(params,lr=recipe['lr'],weight_decay=recipe['weight_decay'])
  # Match v4 training.py: cap per-stage warmup at 10% of that stage.
  stage_warmup=min(int(recipe.get('warmup_steps',0)),total//10)
  def schedule(step,total=total,warm=stage_warmup):
   if step<warm:return (step+1)/warm
   progress=(step-warm)/max(1,total-warm)
   return .5*(1+math.cos(math.pi*min(progress,1.)))
  scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
  seed_all(a.seed+100000*si+rank)
  if resume and si==resume['stage_index']:
   optimizer.load_state_dict(resume['optimizer_state']);scheduler.load_state_dict(resume['scheduler_state'])
   rng=resume['rank_rng'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],device) if not a.cpu else None
  del resume;resume=None
  def save():
   states=[None]*world
   rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=None if a.cpu else torch.cuda.get_rng_state(device))
   if world>1:dist.all_gather_object(states,rng)
   else:states[0]=rng
   # In addition to DDP, explicitly check synchronized model parameters at saves.
   error=torch.zeros((),device=device)
   for v in params:
    ref=v.detach().clone();broadcast(ref);error=torch.maximum(error,(ref-v.detach()).abs().max())
   reduce(error,op=dist.ReduceOp.MAX);assert error.item()<1e-6
   if rank==0:
    checkpoint=dict(format_version=4,model_config=policy.config,model_state=policy.checkpoint_state_dict(),training_stage=stage,stage_index=si,step=step,calibration=joint.calibration,args=vars(a),world_size=world,data_manifest_sha256=data_hash,recipe_sha256=recipe_hash,recipe=recipe,optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),rank_rng=states,parameter_sync_error=error.item(),sampler='global weighted draw per step; deterministic seed; rank slicing')
    tmp=out/'resume.tmp';torch.save(checkpoint,tmp);tmp.replace(out/'resume.pt')
    if step==total:
     target=out/f'adaspevla_franka_{recipe["name"]}_{stage}.pt';torch.save(checkpoint,target)
     if stage=='timing':torch.save(checkpoint,out/f'adaspevla_franka_{recipe["name"]}_latest.pt')
   barrier()
  started=time.perf_counter();first=step;loss_sum=0.;count=0
  write_status(status='training',stage=stage,step=step,total_steps=total)
  while step<total:
   # Stateless global draw: exact same data resumes at the same optimizer step.
   g=torch.Generator().manual_seed(a.seed+si*100000000+step)
   indices=torch.multinomial(joint.sampler.weights,a.local_batch*world,replacement=True,generator=g).tolist()
   local_indices=indices[rank*a.local_batch:(rank+1)*a.local_batch]
   batch={k:move(v) for k,v in collate_windows([joint[i] for i in local_indices]).items()}
   optimizer.zero_grad(set_to_none=True);loss=ddp(batch)
   finite=torch.isfinite(loss).to(torch.int32);reduce(finite,op=dist.ReduceOp.MIN)
   if not finite.item():raise RuntimeError('Nonfinite loss; stopping all ranks')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,recipe['gradient_clip'])
   finite=torch.isfinite(norm).to(torch.int32);reduce(finite,op=dist.ReduceOp.MIN)
   if not finite.item():raise RuntimeError('Nonfinite gradient; stopping all ranks')
   optimizer.step();scheduler.step();step+=1;updates+=1
   mean=loss.detach().clone();reduce(mean);loss_sum+=mean.item()/world;count+=1
   stopping=a.max_updates is not None and updates>=a.max_updates
   if step%a.log_every==0 or step==total or stopping:
    torch.cuda.synchronize() if not a.cpu else None;seconds=(time.perf_counter()-started)/(step-first)
    write_status(status='training',stage=stage,step=step,total_steps=total,loss=loss_sum/count,seconds_per_step=seconds,lr=optimizer.param_groups[0]['lr'])
    if rank==0:
     row=dict(stage=stage,step=step,loss=loss_sum/count,seconds_per_step=seconds,lr=optimizer.param_groups[0]['lr'])
     with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
     print(row,flush=True)
    loss_sum=0.;count=0
   if step%a.save_every==0 or step==total or stopping:save()
   if stopping:
    write_status(status='stopped_for_resume',stage=stage,step=step,total_steps=total);finish();return
  del ddp,wrapper,optimizer,scheduler
 write_status(status='completed',stage='timing',step=a.timing_steps,total_steps=a.timing_steps)
 finish()

if __name__=='__main__':main()
