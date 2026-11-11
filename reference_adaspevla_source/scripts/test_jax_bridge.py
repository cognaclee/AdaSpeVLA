import json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp,torch
from adaspevla.models.jax_pi05 import _VJP,to_jax
class Tiny:
 def __init__(self):
  self.params={'w':jnp.asarray([[.3,-.7],[.2,.8]],dtype=jnp.float32)};self.grads=None
  self.forward_jit=jax.jit(lambda p,x,ids,valid:(x@p['w'],(x@p['w']).mean(0,keepdims=True)))
 def backward(self,inputs,gt,gx):
  def loss(p):
   t,x=self.forward_jit(p,*inputs);return (t*gt).sum()+(x*gx).sum()
  g=jax.grad(loss)(self.params);self.grads=g if self.grads is None else jax.tree.map(jnp.add,self.grads,g)
x=torch.tensor([[1.,2.],[-1.,3.]],device='cuda');ids=torch.zeros((2,1),device='cuda',dtype=torch.int32);valid=ids==0
e=Tiny();s=torch.zeros((),device='cuda',requires_grad=True);t,z=_VJP.apply(s,x,ids,valid,e);loss=t.square().mean()+z.square().mean();loss.backward()
w=torch.tensor([[.3,-.7],[.2,.8]],device='cuda',requires_grad=True);y=x@w;expected=y.square().mean()+y.mean(0,keepdim=True).square().mean();expected.backward();np.testing.assert_allclose(np.array(e.grads['w']),w.grad.cpu().numpy(),rtol=1e-5,atol=1e-6)
# Test accumulation with a per-example objective, matching the trainer's mean weighting.
e=Tiny()
for i in range(2):
 t,z=_VJP.apply(torch.zeros((),device='cuda',requires_grad=True),x[i:i+1],ids[i:i+1],valid[i:i+1],e);((t.square().mean()+z.square().mean())/2).backward()
w.grad=None;y=x@w;(2*y.square().mean()).backward();np.testing.assert_allclose(np.array(e.grads['w']),w.grad.cpu().numpy(),rtol=1e-5,atol=1e-6)
print('PASSED: JAX VJP bridge matches Torch reference; microbatch gradients match full-batch objective')
