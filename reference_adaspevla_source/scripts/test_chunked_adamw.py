import copy
import torch
from adaspevla.models.jax_pi05 import chunked_adamw

torch.set_num_threads(4);torch.manual_seed(73)
for size,chunk in [(37,11),(2097153,1048576)]:
 a=torch.nn.Parameter(torch.randn(size));b=torch.nn.Parameter(a.detach().clone())
 ref=torch.optim.AdamW([a],lr=2e-4,weight_decay=1e-4,foreach=False)
 state={}
 for step in range(5):
  grad=torch.randn(size) if step!=2 else torch.zeros(size)
  clip=.123;lr=2e-4*(step+1)/5
  a.grad=grad*clip;ref.param_groups[0]['lr']=lr;ref.step();ref.zero_grad(set_to_none=True)
  chunked_adamw(b,grad,state,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=1e-4,clip_factor=clip,chunk_elements=chunk)
  torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
  for key in ['step','exp_avg','exp_avg_sq']:torch.testing.assert_close(state[key],ref.state[a][key],atol=1e-7,rtol=1e-6)
  if step==2:state=copy.deepcopy(ref.state_dict()['state'][0])
 print('PASS',size,'CPU AdamW updates/moments and resume compatibility')
# Exercise device-to-host streaming too.
a=torch.nn.Parameter(torch.randn(257));b=torch.nn.Parameter(a.detach().clone());g=torch.randn(257,device='cuda')
ref=torch.optim.AdamW([a],lr=2e-4,weight_decay=1e-4,foreach=False);a.grad=g.cpu();ref.step()
chunked_adamw(b,g,{},lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4,chunk_elements=31)
torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6);print('PASS GPU-to-CPU chunk streaming')
