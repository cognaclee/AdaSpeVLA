import dataclasses
import flax.linen as nn
import jax,jax.numpy as jnp,numpy as np
from openpi.models import gemma
from adaspevla.models.jax_pi05 import _front_layers

spec=dataclasses.replace(gemma.get_config('dummy'),depth=6)
full=gemma.Module(configs=[spec],embed_dtype='float32',adarms=True)
front=gemma.Module(configs=[dataclasses.replace(spec,depth=2)],embed_dtype='float32',adarms=True)
back=gemma.Module(configs=[dataclasses.replace(spec,depth=4)],embed_dtype='float32',adarms=True)
params=nn.Module.init(full,jax.random.key(2),[jnp.ones((1,4,spec.width))],jnp.arange(4)[None,:],jnp.ones((1,4,4),dtype=bool))['params']
first=jax.tree.map(lambda x:x[:2],params['layers']);last=jax.tree.map(lambda x:x[2:],params['layers'])
x=jax.random.normal(jax.random.key(3),(1,4,spec.width));pos=jnp.arange(4)[None,:];mask=jnp.ones((1,4,4),dtype=bool)
expected=full.apply({'params':params},[x],pos,mask)[0][0]
frozen=front.apply({'params':{'layers':first}},x,pos,mask,method=_front_layers)
actual=back.apply({'params':{'layers':last,'final_norm':params['final_norm']}},[frozen],pos,mask)[0][0]
np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=1e-6,atol=1e-6)
def full_loss(p):return jnp.sum(jnp.square(full.apply({'params':p},[x],pos,mask)[0][0]))
def part_loss(p):return jnp.sum(jnp.square(back.apply({'params':{'layers':p,'final_norm':params['final_norm']}},[jax.lax.stop_gradient(frozen)],pos,mask)[0][0]))
a=jax.grad(full_loss)(params)['layers'];b=jax.grad(part_loss)(last)
for p,q in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
 np.testing.assert_allclose(np.asarray(p[2:]),np.asarray(q),rtol=3e-5,atol=3e-5)
 assert all(float(jnp.linalg.norm(q[i]))>0 for i in range(4))
print('PASS split Gemma output matches full model; all four layer gradients match full-model gradients')
