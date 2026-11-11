"""Official JAX pi05 VLM, differentiated through a Torch autograd VJP boundary.

The vision and language weights remain JAX arrays, all participating in the VJP.
FP32 AdamW placement is selected by ADASPEVLA_OPTIMIZER_DEVICE (CPU default). The native
pi05 action expert is excluded because AdaSpeVLA retains its own action heads.
"""
import os
import gc
import ctypes
import math
import dataclasses
from pathlib import Path
import numpy as np
import torch
from torch import nn
import jax
import jax.numpy as jnp
from flax import traverse_util
import orbax.checkpoint as ocp
from openpi.models import gemma, siglip


def restore_vlm(path):
    # OpenPI model.restore_params logic, without importing its native policy.
    with ocp.PyTreeCheckpointer() as ckpt:
        meta=ckpt.metadata(path); item={"params":meta["params"]}
        values=ckpt.restore(path,ocp.args.PyTreeRestore(item=item,restore_args=jax.tree.map(
            lambda _:ocp.ArrayRestoreArgs(restore_type=np.ndarray),item)))["params"]
    flat=traverse_util.flatten_dict(values)
    if all(k[-1]=="value" for k in flat):flat={k[:-1]:v for k,v in flat.items()}
    selected={k[1:]:v for k,v in flat.items() if k[0]=="PaliGemma" and (k[1]=="img" or (k[1]=="llm" and not any(x.endswith("_1") for x in k[2:])))}
    assert selected[("llm","layers","mlp","linear")].shape[0]==18
    assert selected[("img","Transformer","encoderblock","LayerNorm_0","scale")].shape[0]==27
    return selected


def _front_layers(module,embedded,positions,attention):
    # Use the same OpenPI Gemma scan without its final output normalization.
    xs,_=module.layers([embedded],None,positions,attention[:,None,:,:],[None],True)
    return xs[0]


class JaxEngine:
    """Frozen full vision and first14 Gemma blocks; train last4 blocks only."""
    def __init__(self,path):
        values=restore_vlm(str(Path(path).resolve()))
        keys=sorted(k for k in values if len(k)>=3 and k[:2]==("llm","layers"))
        assert len(keys)>0 and all(values[k].shape[0]==18 for k in keys)
        self.keys=keys
        self.masters=[nn.Parameter(torch.from_numpy(np.array(values[k][-4:],dtype=np.float32,copy=True)).to(os.environ.get("ADASPEVLA_OPTIMIZER_DEVICE", "cpu"))) for k in keys]
        frozen_flat={}
        for k,v in values.items():
            if k in keys:
                frozen_flat[k]=jax.device_put(np.ascontiguousarray(v[:14]))
            else:
                frozen_flat[k]=jax.device_put(np.ascontiguousarray(v))
        self.device=jax.devices("gpu")[0]
        self.frozen=traverse_util.unflatten_dict(frozen_flat)
        jax.block_until_ready(self.frozen)
        del values,frozen_flat;gc.collect();ctypes.CDLL(None).malloc_trim(0)
        self.optimizer=None;self.grads=None;self.trainable=True;self.backward_calls=0
        self.sync_params()
        self.vision=siglip.Module(num_classes=2048,variant="So400m/14",pool_type="none",scan=True,dtype_mm="bfloat16")
        spec=gemma.get_config("gemma_2b")
        self.front=gemma.Module(configs=[dataclasses.replace(spec,depth=14)],embed_dtype="bfloat16",adarms=True)
        self.back=gemma.Module(configs=[dataclasses.replace(spec,depth=4)],embed_dtype="bfloat16",adarms=True)
        jf=jax.jit(self._forward)
        jg=jax.jit(jax.grad(self._contract,argnums=0))
        self.forward_jit=lambda params,*args:jf(params,self.frozen,*args)
        self.grad_jit=lambda params,*args:jg(params,self.frozen,*args)
        self.add_jit=jax.jit(lambda x,y:jax.tree.map(jnp.add,x,y),donate_argnums=(0,))
        self.norm_jit=jax.jit(lambda g:sum(jnp.vdot(x,x).real for x in jax.tree.leaves(g)))

    def sync_params(self):
        if hasattr(self,"params"):
            jax.block_until_ready(self.params);del self.params
        self.params=traverse_util.unflatten_dict({k:(to_jax(p.detach().clone()) if p.is_cuda else jax.device_put(p.detach().numpy(),self.device)) for k,p in zip(self.keys,self.masters)})
        jax.block_until_ready(self.params)

    def _forward(self,params,frozen,pixels,ids,valid):
        b,v,h,w,c=pixels.shape
        image_tokens,_=self.vision.apply({"params":frozen["img"]},pixels.reshape(b*v,h,w,c),train=False)
        n=image_tokens.shape[1];image_tokens=image_tokens.reshape(b,v*n,2048)
        text_tokens=self.front.apply({"params":frozen["llm"]},ids,method=self.front.embed)
        x=jnp.concatenate([image_tokens,text_tokens],axis=1)
        mask=jnp.concatenate([jnp.ones((b,v*n),dtype=bool),valid],axis=1)
        positions=jnp.cumsum(mask,axis=1)-1;attention=mask[:,:,None]&mask[:,None,:]
        hidden=self.front.apply({"params":frozen["llm"]},x,positions,attention,method=_front_layers)
        hidden=jax.lax.stop_gradient(hidden)
        back_params={"layers":params["llm"]["layers"],"final_norm":frozen["llm"]["final_norm"]}
        hidden,_=self.back.apply({"params":back_params},[hidden],positions,attention)
        hidden=hidden[0].astype(jnp.float32)
        text=(hidden[:,v*n:]*valid[:,:,None]).sum(1)/jnp.maximum(valid.sum(1,keepdims=True),1)
        return hidden[:,:v*n].reshape(b,v,n,2048),text

    def _contract(self,params,frozen,pixels,ids,valid,grad_tokens,grad_text):
        tokens,text=self._forward(params,frozen,pixels,ids,valid)
        return jnp.sum(tokens*grad_tokens)+jnp.sum(text*grad_text)

    def backward(self,inputs,grad_tokens,grad_text):
        g=self.grad_jit(self.params,*inputs,grad_tokens,grad_text)
        self.grads=g if self.grads is None else self.add_jit(self.grads,g)
        jax.block_until_ready(self.grads);self.backward_calls+=1

    def init_optimizer(self,lr,weight_decay):
        self.optimizer=torch.optim.AdamW(self.masters,lr=lr,weight_decay=weight_decay,foreach=False)

    def grad_norm_sq(self):
        if self.grads is None:raise RuntimeError("Missing JAX backbone gradients")
        return float(self.norm_jit(self.grads))

    def step(self,lr,clip_factor):
        if self.optimizer is None:raise RuntimeError("Missing backbone optimizer")
        flat=traverse_util.flatten_dict(self.grads)
        self.optimizer.param_groups[0]["lr"]=lr;group=self.optimizer.param_groups[0]
        with torch.no_grad():
            for k,p in zip(self.keys,self.masters):
                grad=torch.from_dlpack(flat.pop(k)).reshape(-1)
                chunked_adamw(p,grad,self.optimizer.state[p],lr=lr,betas=group['betas'],eps=group['eps'],weight_decay=group['weight_decay'],clip_factor=clip_factor)
                del grad
        self.grads=None;del flat;self.sync_params();gc.collect();ctypes.CDLL(None).malloc_trim(0)

    def gradient_report(self):
        flat=traverse_util.flatten_dict(self.grads)
        return {"/".join(k):dict(shape=list(g.shape),finite=bool(jnp.all(jnp.isfinite(g))),norm=float(jnp.linalg.norm(g)),layer_norms=[float(jnp.linalg.norm(g[i])) for i in range(4)],numel=int(g.size)) for k,g in flat.items()}

    def state_dict(self,optimizer=True):
        return dict(keys=self.keys,weights=[p.detach() for p in self.masters],optimizer=self.optimizer.state_dict() if optimizer and self.optimizer else None,freeze_contract="pi05_base_vision27_lm1to14_finalnorm_frozen_lm15to18_trainable")

    def load_state_dict(self,state):
        assert [tuple(k) for k in state["keys"]]==self.keys
        assert state.get("freeze_contract")=="pi05_base_vision27_lm1to14_finalnorm_frozen_lm15to18_trainable"
        with torch.no_grad():
            for p,w in zip(self.masters,state["weights"]):p.copy_(w)
        self.sync_params()
        if state.get("optimizer") is not None:
            assert self.optimizer is not None
            self.optimizer.load_state_dict(state["optimizer"])


@torch.no_grad()
def chunked_adamw(parameter,gradient,state,*,lr,betas,eps,weight_decay,clip_factor=1.,chunk_elements=1048576):
    """FP32 AdamW identical equations to torch AdamW(foreach=False), bounded scratch."""
    assert parameter.dtype==torch.float32
    if not state:
        state['step']=torch.tensor(0.,dtype=torch.float32)
        state['exp_avg']=torch.zeros_like(parameter)
        state['exp_avg_sq']=torch.zeros_like(parameter)
    state['step'].add_(1)
    step=float(state['step']);beta1,beta2=betas
    bc1=1-beta1**step;bc2_sqrt=math.sqrt(1-beta2**step)
    values=parameter.view(-1);moment=state['exp_avg'].view(-1);variance=state['exp_avg_sq'].view(-1)
    for start in range(0,values.numel(),chunk_elements):
        end=min(start+chunk_elements,values.numel())
        # copy=True also makes CPU-only reference tests avoid modifying their input.
        g=gradient[start:end].to(device=parameter.device,dtype=torch.float32,copy=True).mul_(clip_factor)
        v=values[start:end];m=moment[start:end];q=variance[start:end]
        v.mul_(1-lr*weight_decay)
        m.lerp_(g,1-beta1);q.mul_(beta2).addcmul_(g,g,value=1-beta2)
        denominator=q.sqrt().div_(bc2_sqrt).add_(eps)
        v.addcdiv_(m,denominator,value=-lr/bc1)



def to_jax(t):
    value=t.detach().contiguous()
    # XLA GEMM requires aligned storage; contiguous slices can retain an offset.
    if value.data_ptr() % 256:
        value=value.clone()
    return jax.dlpack.from_dlpack(value)


class _VJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sentinel, pixels, ids, valid, engine):
        inputs=(to_jax(pixels),to_jax(ids),to_jax(valid))
        tokens,text=engine.forward_jit(engine.params,*inputs)
        ctx.engine=engine;ctx.inputs=inputs
        return torch.from_dlpack(tokens),torch.from_dlpack(text)

    @staticmethod
    def backward(ctx,grad_tokens,grad_text):
        ctx.engine.backward(ctx.inputs,to_jax(grad_tokens),to_jax(grad_text))
        return None,None,None,None,None


class JaxPi05Encoder(nn.Module):
    hidden_size=2048
    image_size=(224,224)
    def __init__(self):
        super().__init__()
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224",local_files_only=True)
        self.engine=JaxEngine(os.environ["ADASPEVLA_JAX_PARAMS"])
        self.ids_cache={}

    def forward(self,pixels,prompts,num_views):
        b=len(prompts);seq=[]
        for prompt in prompts:
            if prompt not in self.ids_cache:
                text="Task: {};\nAction: ".format(prompt.strip().replace("_"," ").replace("\n"," "))
                self.ids_cache[prompt]=self.tokenizer(text)["input_ids"][:200]
            seq.append(self.ids_cache[prompt])
        width=max(map(len,seq));ids=torch.zeros((b,width),device=pixels.device,dtype=torch.int32);valid=torch.zeros_like(ids,dtype=torch.bool)
        for i,s in enumerate(seq):ids[i,:len(s)]=torch.tensor(s,device=pixels.device);valid[i,:len(s)]=True
        pixels=pixels.permute(0,2,3,1).reshape(b,num_views,224,224,3).contiguous()
        if torch.is_grad_enabled() and self.engine.trainable:
            sentinel=torch.zeros((),device=pixels.device,requires_grad=True)
            return _VJP.apply(sentinel,pixels,ids,valid,self.engine)
        tokens,text=self.engine.forward_jit(self.engine.params,to_jax(pixels),to_jax(ids),to_jax(valid))
        return torch.from_dlpack(tokens),torch.from_dlpack(text)
