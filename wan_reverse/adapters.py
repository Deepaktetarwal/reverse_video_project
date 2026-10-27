"""FP32 trainable LoRA matrices around frozen Wan attention projections."""
import math
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__()
        self.base=base
        self.base.requires_grad_(False)
        self.rank=rank; self.alpha=alpha; self.scale=alpha/rank
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
    def forward(self,x):
        base=self.base(x)
        # Parameters and optimizer master values remain FP32 on a T4.
        with torch.autocast(device_type=x.device.type,enabled=False):
            residual=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.lora_A),self.lora_B)*self.scale
        return base+residual.to(base.dtype)

def inject(model,rank,alpha):
    model.requires_grad_(False)
    names=[]
    for name,module in list(model.named_modules()):
        if isinstance(module,nn.Linear) and any(name.endswith(s) for s in ('.to_q','.to_k','.to_v','.to_out.0')):
            parent_name,child=name.rsplit('.',1)
            parent=model.get_submodule(parent_name)
            setattr(parent,child,LoRALinear(module,rank,alpha))
            names.append(name)
    if not names: raise ValueError('No Wan attention targets matched')
    return names

def adapter_state(model):
    return {n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if n.endswith(('lora_A','lora_B'))}

def load_adapter(model,state):
    expected={n:p for n,p in model.named_parameters() if n.endswith(('lora_A','lora_B'))}
    if set(expected)!=set(state): raise ValueError('Adapter keys do not match model/rank')
    with torch.no_grad():
        for n,p in expected.items():
            if p.shape!=state[n].shape: raise ValueError('Adapter shape mismatch')
            p.copy_(state[n].to(p.device,p.dtype))

def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]
