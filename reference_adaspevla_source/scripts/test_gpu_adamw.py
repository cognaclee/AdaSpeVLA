import copy
import torch
from adaspevla.models.jax_pi05 import chunked_adamw, to_jax
import jax

torch.manual_seed(73)
a=torch.nn.Parameter(torch.randn(2097153))
ref=torch.optim.AdamW([a],lr=2e-4,weight_decay=1e-4,foreach=False)
a.grad=torch.randn_like(a);ref.step();ref.zero_grad(set_to_none=True)
b=torch.nn.Parameter(a.detach().cuda())
opt=torch.optim.AdamW([b],lr=2e-4,weight_decay=1e-4,foreach=False)
opt.load_state_dict(copy.deepcopy(ref.state_dict()))
assert opt.state[b]['exp_avg'].is_cuda
for i in range(5):
 g=torch.randn_like(a);a.grad=g*.123;ref.step();ref.zero_grad(set_to_none=True)
 chunked_adamw(b,g.cuda(),opt.state[b],lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4,clip_factor=.123)
 torch.testing.assert_close(a,b.cpu(),atol=1e-7,rtol=1e-6)
 for key in ['exp_avg','exp_avg_sq']:
  torch.testing.assert_close(ref.state[a][key],opt.state[b][key].cpu(),atol=1e-7,rtol=1e-6)
x=to_jax(b.detach().clone());jax.block_until_ready(x)
torch.testing.assert_close(torch.from_dlpack(x),b)
print('PASS CPU checkpoint to CUDA AdamW: five updates, moments, and JAX transfer')
