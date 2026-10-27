import random
from pathlib import Path
import torch
from safetensors.torch import load_file
from .common import *
from .models import predict
from .losses import interpolate,losses
from .adapters import trainable,adapter_state,load_adapter
from .conditioning import MODE,student_embedding

class CachedBatch:
    def __init__(self,c,spec):
        self.c=c
        d=load_file(str(tensor_path(c,'prepared',spec['prepared'])))
        k=load_file(str(tensor_path(c,'kd_cache',spec['file'])))
        self.zr=d['zr'].to('cuda',torch.float32)
        self.noise=k['noise'].to('cuda',torch.float32)
        self.sigma=float(k['sigma'])
        self.kd=k['kd'].to('cuda',torch.float32)
        self.emb=student_embedding(d,dropped=bool(k['drop'])).to('cuda')
        self.xr=interpolate(self.zr,self.noise,self.sigma)
        self.target=self.noise-self.zr
    def loss(self,model):
        pred=predict(model,self.xr,self.sigma,self.emb,self.c)
        return losses(pred,self.target,self.kd,self.zr,self.xr,self.sigma,self.c)

def validation(model,c,specs):
    model.eval(); values=[]
    with torch.no_grad():
        for spec in specs:
            _,v=CachedBatch(c,spec).loss(model)
            values.append({k:float(x) for k,x in v.items()})
    model.train()
    return {k:sum(r[k] for r in values)/len(values) for k in values[0]} if values else {}

def checkpoint(root,model,opt,scaler,c,step,best_val,sampler):
    root=Path(root)/'checkpoints'/f'step_{step:06d}'
    root.mkdir(parents=True,exist_ok=True)
    if (root/'metadata.json').exists(): raise ValueError('Refusing to overwrite an existing committed checkpoint')
    adapter=root/'adapter.safetensors'
    save_tensors(adapter,adapter_state(model))
    state={'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'step':step,'best_val':best_val,'sampler_state':sampler.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    tmp=root/'training.pt.tmp'; torch.save(state,tmp); tmp.replace(root/'training.pt')
    metadata={'conditioning':MODE,'step':step,'train_signature':train_signature(c),'model_id':c['model_id'],'revision':c['revision'],'rank':c['rank'],'lora_alpha':c['lora_alpha'],'adapter_sha256':sha256(adapter),'state_sha256':sha256(root/'training.pt'),'best_validation_flow':best_val if __import__('math').isfinite(best_val) else None,'config':c}
    write_json(root/'metadata.json',metadata)
    write_json(Path(c['run_dir'])/'latest.json',{'checkpoint':str(root)})
    return root

def resolve_checkpoint(c,name='best'):
    if name in ('best','latest'):
        return Path(read_json(Path(c['run_dir'])/(name+'.json'))['checkpoint'])
    return Path(name)

def verify_checkpoint(c,path,training=False):
    path=Path(path); m=read_json(path/'metadata.json')
    if m.get('conditioning')!=MODE: raise ValueError('Checkpoint used another conditioning task. Train a fresh original-prompt adapter; old reverse-caption weights cannot be reused.')
    if m['model_id']!=c['model_id'] or m['revision']!=c['revision'] or m['rank']!=c['rank'] or m['lora_alpha']!=c['lora_alpha']: raise ValueError('Checkpoint/base-model mismatch')
    if sha256(path/'adapter.safetensors')!=m['adapter_sha256']: raise ValueError('Corrupt adapter')
    if training:
        if m['train_signature']!=train_signature(c): raise ValueError('Resume signature mismatch; use a new run_dir for an ablation')
        if sha256(path/'training.pt')!=m['state_sha256']: raise ValueError('Corrupt training state')
    return m
