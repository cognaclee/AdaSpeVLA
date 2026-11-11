"""Frozen GR00T N1.7 robot-pretrained VLM with explicit feature depth.

Uses Qwen3-VL's native multimodal RoPE and DeepStack; no N1.5 weight mapping.
The common AdaSpeVLA recipe uses pad/full-view 256px input. Image token count
is native (64 per view), not artificially matched to SigLIP's 256 tokens.
"""
import os
from pathlib import Path
import torch
from torch import nn

REPO='nvidia/GR00T-N1.7-3B'
REVISION='2fc962b973bccdd5d8ce4f67cc63b264d6886495'
PROCESSOR_REPO='nvidia/Cosmos-Reason2-2B'
PROCESSOR_REVISION='9ce19a195e423419c349abfc86fd07178b230561'

class Gr00t17Encoder(nn.Module):
    def __init__(self,lm_layers=None):
        super().__init__()
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig,AutoProcessor,Qwen3VLModel
        from safetensors import safe_open
        depth = 16 if lm_layers is None else int(lm_layers)
        if not 1 <= depth <= 16:
            raise ValueError('N1.7 lm_layers must be between 1 and 16')
        self.lm_layers = depth
        path=Path(snapshot_download(REPO,revision=REVISION,local_files_only=True))
        proc=Path(snapshot_download(PROCESSOR_REPO,revision=PROCESSOR_REVISION,local_files_only=True))
        config=AutoConfig.from_pretrained(proc,local_files_only=True)
        # Instantiate only the requested prefix: later decoder weights are never
        # loaded. Native visual DeepStack injections and final norm are retained.
        config.text_config.num_hidden_layers=depth
        config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
        with torch.device('meta'):self.model=Qwen3VLModel(config)
        expected=self.model.state_dict();state={}
        prefix='backbone.model.model.'
        for shard in sorted(path.glob('*.safetensors')):
            with safe_open(shard,framework='pt') as f:
                for key in f.keys():
                    if key.startswith(prefix) and key[len(prefix):] in expected:
                        state[key[len(prefix):]]=f.get_tensor(key).to(torch.bfloat16)
        missing=set(expected)-set(state)
        if missing:raise RuntimeError(f'N1.7 missing backbone tensors: {sorted(missing)[:10]}')
        self.model.load_state_dict(state,strict=True,assign=True)
        # Rotary frequency buffers are non-persistent; initialize from the same
        # native module constructors rather than leaving meta buffers behind.
        for module in self.model.modules():
            for name,b in list(module.named_buffers(recurse=False)):
                if b.is_meta:
                    if name=='inv_freq' and hasattr(module,'rope_init_fn'):
                        inv,scale=module.rope_init_fn(module.config,device=torch.device('cpu'))
                        module.register_buffer(name,inv,persistent=False)
                        module.original_inv_freq=inv
                        module.attention_scaling=scale
                    elif name=='inv_freq' and type(module).__name__=='Qwen3VLVisionRotaryEmbedding':
                        dim=config.vision_config.hidden_size//config.vision_config.num_heads//2
                        module.register_buffer(name,type(module)(dim).inv_freq,persistent=False)
                    else:raise RuntimeError(f'Uninitialized buffer {type(module).__name__}.{name}')
        self.processor=AutoProcessor.from_pretrained(proc,local_files_only=True)
        self.hidden_size=2048;self.image_size=(256,256);self.tokens_per_image=64
        self.requires_grad_(False);self.eval()
    def forward(self,pixels,prompts,num_views):
        device=next(self.model.parameters()).device
        texts=[]
        for prompt in prompts:
            content=[{'type':'image'} for _ in range(num_views)]+[{'type':'text','text':prompt}]
            texts.append(self.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True))
        inputs=self.processor(text=texts,images=list(pixels.detach().cpu()),padding=True,return_tensors='pt',do_rescale=False,do_resize=False)
        inputs={k:v.to(device) for k,v in inputs.items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
        inputs['pixel_values']=inputs['pixel_values'].to(torch.bfloat16)
        output=self.model(**inputs,use_cache=False,return_dict=True).last_hidden_state
        mask=inputs['input_ids']==self.model.config.image_token_id
        assert torch.all(mask.sum(1)==num_views*self.tokens_per_image)
        images=torch.stack([o[m].reshape(num_views,self.tokens_per_image,self.hidden_size) for o,m in zip(output,mask)])
        # Pool attended textual context (chat template + instruction), excluding
        # image tokens. This choice is fixed across train, reload and deployment.
        tm=inputs['attention_mask'].bool()&~mask
        text=(output*tm.unsqueeze(-1)).sum(1)/tm.sum(1,keepdim=True).clamp_min(1)
        return images,text
