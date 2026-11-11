"""Single-GPU JAX VLM + Torch AdaSpeVLA training, exact effective-batch accumulation."""
import os,time,json,math,hashlib,argparse,random,signal,resource,gc,ctypes
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import torch
import jax
from adaspevla.models.adaspe_vla_policy import AdaSpeVLAPolicy
from adaspevla.datasets.franka_training import load_franka_training,policy_config
from adaspevla.datasets.bspline_dataset import collate_windows


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--recipe',required=True);ap.add_argument('--preparation-report',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--action-steps',type=int);ap.add_argument('--timing-steps',type=int);ap.add_argument('--max-updates',type=int);ap.add_argument('--resume',action='store_true');ap.add_argument('--audit',action='store_true');ap.add_argument('--batch',type=int)
    ap.add_argument('--checkpoint-every',type=int,default=0);ap.add_argument('--resource-floor-gb',type=float,default=12.)
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    recipe=json.loads(Path(a.recipe).read_text());rh=hashlib.sha256(Path(a.recipe).read_bytes()).hexdigest()
    prep=json.loads(Path(a.preparation_report).read_text());assert prep['status']=='passed' and prep['recipe_sha256']==rh
    batch=a.batch or recipe['local_batch'];micro=recipe['micro_batch'];assert batch%micro==0
    totals=[a.action_steps or recipe['action_steps'],a.timing_steps or recipe['timing_steps']]
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True;seed_all(recipe['seed'])
    def resource_stats():
        values={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapFree:'))}
        rss=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))
        return dict(rss_gb=rss/1048576,host_available_gb=values['MemAvailable']/1048576,swap_free_gb=values['SwapFree']/1048576)
    def status(**kw):
        v=dict(optimizer_device=os.environ.get("ADASPEVLA_OPTIMIZER_DEVICE","cpu"),updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),batch=batch,micro_batch=micro,gradient_accumulation=batch//micro,**(resource_stats()|kw))
        p=out/'status.tmp';p.write_text(json.dumps(v,indent=2)+'\n');p.replace(out/'status.json')
    status(status='loading_data')
    data,dh=load_franka_training(a.data_dir,recipe['task_names'],recipe);assert dh==prep['data_manifest_sha256']
    status(status='loading_jax_backbone')
    if a.resume:os.environ['ADASPEVLA_JAX_RESUME']=str((out/'resume.pt').resolve())
    model=AdaSpeVLAPolicy(**policy_config(data,recipe)).cuda();engine=model.vlm.vision_encoder.engine
    counts=dict(backbone_trainable_parameters=sum(p.numel() for p in engine.masters),backbone_total_parameters=2923335408,head_parameters=sum(p.numel() for p in model.parameters()),backbone_arrays=len(engine.keys),language_layers=18,vision_layers=27,trainable_language_layers=[15,16,17,18],frozen_vision_layers=27)
    (out/'parameter_counts.json').write_text(json.dumps(counts,indent=2)+'\n');print(counts,flush=True)
    resume=None
    if a.resume:
        resume=torch.load(out/'resume.pt',map_location='cpu',weights_only=False,mmap=True)
        assert resume['recipe_sha256']==rh and resume['data_manifest_sha256']==dh and resume['batch']==batch and resume['totals']==totals
        model.load_checkpoint_state(resume['model_state'])
    elif (out/'resume.pt').exists():raise RuntimeError('Existing resume.pt requires --resume')
    stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    def move(v):
        if isinstance(v,torch.Tensor):return v.cuda()
        if isinstance(v,list):return [move(x) for x in v]
        return v
    updates=0
    for si,(stage,total) in enumerate(zip(['action','timing'],totals)):
        if resume and si<resume['stage_index']:continue
        step=resume['step'] if resume and si==resume['stage_index'] else 0
        model.set_training_stage(stage);model.train();data.include_diagnostic_targets=stage!='timing'
        params=[p for p in model.parameters() if p.requires_grad]
        if stage=='action':
            assert engine.trainable and model.vlm.trainable_lm and model.vlm.trainable_vision_layers==0
            engine.init_optimizer(recipe['lr'],recipe['weight_decay'])
        else:
            engine.optimizer=None;assert not engine.trainable
            assert all(n.startswith('freq_head.') for n,p in model.named_parameters() if p.requires_grad)
        opt=torch.optim.AdamW(params,lr=recipe['lr'],weight_decay=recipe['weight_decay'],foreach=False)
        warm=min(recipe['warmup_steps'],total//10)
        def schedule(n):
            if n<warm:return (n+1)/warm
            return .5*(1+math.cos(math.pi*min((n-warm)/max(1,total-warm),1.)))
        sched=torch.optim.lr_scheduler.LambdaLR(opt,schedule);seed_all(recipe['seed']+100000*si)
        if resume and si==resume['stage_index']:
            engine.load_state_dict(resume['jax_backbone']);assert all(torch.equal(p.detach(),w.to(p.device)) for p,w in zip(engine.masters,resume['jax_backbone']['weights']));print('Restored full JAX backbone weights exactly',flush=True);opt.load_state_dict(resume['optimizer_state']);sched.load_state_dict(resume['scheduler_state'])
            rng=resume['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
            del resume;resume=None;gc.collect();ctypes.CDLL(None).malloc_trim(0)
        def save(final=False):
            status(status='saving',stage=stage,step=step,total_steps=total)
            obj=dict(format_version=5,model_config=model.config,model_state=model.checkpoint_state_dict(),jax_backbone=engine.state_dict(),training_stage=stage,stage_index=si,step=step,recipe=recipe,recipe_sha256=rh,data_manifest_sha256=dh,calibration=data.calibration,batch=batch,totals=totals,optimizer_state=opt.state_dict(),scheduler_state=sched.state_dict(),rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
            tmp=out/'resume.tmp';torch.save(obj,tmp);tmp.replace(out/'resume.pt')
            if final:
                obj['jax_backbone']=engine.state_dict(optimizer=False);obj.pop('optimizer_state');obj.pop('scheduler_state')
                target=out/f'adaspevla_franka_{recipe["name"]}_{stage}.pt';tmp=target.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(target)
                if stage=='timing':
                    latest=out/f'adaspevla_franka_{recipe["name"]}_latest.pt'
                    if latest.exists():latest.unlink()
                    os.link(target,latest)
        started=time.perf_counter();first=step;last_saved=time.monotonic()
        if step>=total:continue
        while step<total:
            iteration_started=time.perf_counter()
            initial_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
            indices=torch.multinomial(data.sampler.weights,batch,replacement=True,generator=torch.Generator().manual_seed(recipe['seed']+si*100000000+step)).tolist()
            opt.zero_grad(set_to_none=True);engine.grads=None;loss_sum=0.
            status(status='training',stage=stage,step=step,total_steps=total,micro_step=0)
            for offset in range(0,batch,micro):
                if resource_stats()['host_available_gb']<a.resource_floor_gb:
                    opt.zero_grad(set_to_none=True);engine.grads=None;gc.collect();ctypes.CDLL(None).malloc_trim(0)
                    random.setstate(initial_rng[0]);np.random.set_state(initial_rng[1]);torch.set_rng_state(initial_rng[2]);torch.cuda.set_rng_state(initial_rng[3])
                    save();status(status='resource_guard_stopped',stage=stage,step=step,total_steps=total)
                    print('Stopped safely: low host memory; incomplete update discarded and RNG restored',flush=True)
                    raise SystemExit(75)
                samples={k:move(v) for k,v in collate_windows([data[i] for i in indices[offset:offset+micro]]).items()}
                loss,_=model.forward_loss(samples)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                (loss*(micro/batch)).backward();loss_sum+=loss.item()*micro/batch
                if a.audit and si==0 and step==0 and offset==0:
                    report=engine.gradient_report();assert all(x['finite'] and all(y>0 for y in x['layer_norms']) for x in report.values())
                    (out/'backbone_gradient_report.json').write_text(json.dumps(report,indent=2)+'\n')
                if offset%8==0:status(status='training',stage=stage,step=step,total_steps=total,micro_step=offset//micro+1)
            head_sq=sum(float(p.grad.float().square().sum()) for p in params if p.grad is not None)
            backbone_sq=engine.grad_norm_sq() if stage=='action' else 0.
            norm=math.sqrt(head_sq+backbone_sq)
            if not math.isfinite(norm):raise RuntimeError('Nonfinite gradient norm')
            factor=min(1.,recipe['gradient_clip']/(norm+1e-6));lr=opt.param_groups[0]['lr']
            for p in params:
                if p.grad is not None:p.grad.mul_(factor)
            before=[p.detach().reshape(-1)[::max(1,p.numel()//17)].clone() for p in engine.masters] if a.audit and stage=='action' and step==0 else None
            if stage=='action':engine.step(lr,factor)
            opt.step();sched.step();step+=1;updates+=1
            if before is not None:
                delta={"/".join(k):float((p.detach().reshape(-1)[::max(1,p.numel()//17)]-b).abs().max()) for k,p,b in zip(engine.keys,engine.masters,before)}
                # Embedding sampled rows may have zero gradient; report all arrays and assert both towers update.
                assert all(v>0 for k,v in delta.items() if k.startswith('llm/layers/'))
                (out/'backbone_update_report.json').write_text(json.dumps(delta,indent=2)+'\n')
            row=dict(stage=stage,step=step,loss=loss_sum,lr=lr,gradient_norm=norm,backbone_gradient_norm=math.sqrt(backbone_sq),seconds_per_step=(time.perf_counter()-started)/(step-first),last_step_seconds=time.perf_counter()-iteration_started,**resource_stats())
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step%recipe['log_every']==0 or a.audit or updates<=10:print(row,flush=True)
            stopping=stop[0] or (a.max_updates and updates>=a.max_updates)
            if step%recipe['save_every']==0 or step==total or stopping or (a.checkpoint_every and step%a.checkpoint_every==0) or time.monotonic()-last_saved>=600:
                save(final=step==total);last_saved=time.monotonic()
            status(status='training',total_steps=total,**row)
            if stopping:status(status='stopped_for_resume',stage=stage,step=step,total_steps=total);return
        del opt,sched
    status(status='completed',stage='timing',step=totals[1],total_steps=totals[1])

if __name__=='__main__':main()
