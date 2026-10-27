import json,random
from pathlib import Path
import torch
from safetensors.torch import load_file
from wan_reverse.common import *
from wan_reverse.conditioning import MODE,student_embedding
from wan_reverse.models import predict
from wan_reverse.adapters import adapter_state,load_adapter,LoRALinear
from .objective import objective
METHOD='fresh-flow-endpoint-order-v3'
KEYS=['method','phase','conditioning','precision','rank','lora_alpha','learning_rate','weight_decay','accumulate','gradient_clip','conditioning_dropout','sigma_min','sigma_max','seed','train_ids','motion_weight','motion_floor','order_weight','order_margin','order_floor','order_sigma_min','order_sigma_max','min_pair_separation','kd_weight','kd_every','validation_draws','validation_seed']

def load_config(path,allow_partial=False):
    c=config(path)
    if c.get('method')!=METHOD:raise ValueError('Use a V3 refinement config.')
    if c['order_margin']<0 or c['order_margin']>=1:raise ValueError('Order margin must be in [0,1).')
    if c['kd_weight']<0 or c['kd_every']<1:raise ValueError('Invalid KD setup')
    meta,data=prep_index(c)
    if not meta['complete'] and not allow_partial:raise ValueError('Finish VAE preparation before training/evaluation. Partial preparation is allowed only for the GPU smoke.')
    known={r['id']:r for r in (manifest(c) if allow_partial else data)}
    if allow_partial and not any(r['id'] in c['train_ids'] and r['split']=='train' for r in data):raise ValueError('Prepare at least one selected training clip for the smoke test.')
    if not c['train_ids'] or any(i not in known or known[i]['split']!='train' for i in c['train_ids']):raise ValueError('Only train-split IDs may be used for training.')
    if c['kd_weight']:
        km,entries=cache_index(c)
        if not km['complete']:raise ValueError('Finish the old teacher cache for the optional KD branch.')
    return c

def signature(c):
    m,_=prep_index(c)
    return fingerprint({'prepared':m['signature'],'training':{k:c[k] for k in KEYS}})

def get_data(c):
    _,data=prep_index(c)
    return data

def read_clip(c,row,device='cuda'):
    # Only original prompt embeddings enter the student.
    p=tensor_path(c,'prepared',row['file'])
    return {k:v.to(device) for k,v in load_file(str(p)).items() if k in ('zf','zr','forward_embeds','negative_embeds')}

def draw(d,c,generator,dropout=True):
    sigma=c['sigma_min']+(c['sigma_max']-c['sigma_min'])*float(torch.rand((),generator=generator))
    noise=torch.randn(d['zr'].shape,generator=generator,dtype=torch.float32).to(d['zr'].device)
    dropped=dropout and float(torch.rand((),generator=generator))<c['conditioning_dropout']
    return sigma,noise,student_embedding(d,dropped)

def loss_on(model,d,sigma,noise,emb,c):
    xr=(1-sigma)*d['zr'].float()+sigma*noise
    pred=predict(model,xr,sigma,emb,c)
    return objective(pred,d['zr'],d['zf'],noise,sigma,c)

def validation(model,c,rows,split='val',compare_base=False):
    model.eval();records=[]
    g=torch.Generator('cpu').manual_seed(c['validation_seed']+{'val':0,'test':10000,'train':20000}[split])
    with torch.no_grad():
        for i,row in enumerate(rows):
            d=read_clip(c,row)
            for j in range(c['validation_draws']):
                sigma,noise,emb=draw(d,c,g,dropout=False)
                xr=(1-sigma)*d['zr'].float()+sigma*noise
                pred=predict(model,xr,sigma,emb,c)
                _,values=objective(pred,d['zr'],d['zf'],noise,sigma,c)
                record={k:float(v) for k,v in values.items()}
                if compare_base and i==0 and j==0:
                    layers=[layer for layer in model.modules() if isinstance(layer,LoRALinear)]
                    scales=[layer.scale for layer in layers]
                    try:
                        for layer in layers:layer.scale=0
                        base=predict(model,xr,sigma,emb,c)
                    finally:
                        for layer,scale in zip(layers,scales):layer.scale=scale
                    record['adapter_relative_velocity_rms']=float((pred-base).square().mean().sqrt()/base.square().mean().sqrt().clamp_min(1e-6))
                records.append(record)
    model.train()
    names=set().union(*(r.keys() for r in records)) if records else set()
    result={k:sum(r[k] for r in records if k in r)/sum(k in r for r in records) for k in names}
    active=[r for r in records if r['order_active']>0]
    result['reverse_endpoint_closer_fraction']=sum(r['order_gap']>0 for r in active)/len(active) if active else None
    result['order_active_draws']=len(active);result['draws']=len(records)
    return result

def save_checkpoint(c,model,opt,scaler,step,best,sampler,noise_generator):
    root=Path(c['run_dir']);p=root/'checkpoints'/f'step_{step:06d}';p.mkdir(parents=True,exist_ok=True)
    if (p/'metadata.json').exists():raise ValueError('Refusing to overwrite a committed checkpoint')
    save_tensors(p/'adapter.safetensors',adapter_state(model))
    state={'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'step':step,'best_val':best,'sampler_state':sampler.getstate(),'noise_rng':noise_generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    tmp=p/'training.pt.tmp';torch.save(state,tmp);tmp.replace(p/'training.pt')
    metadata={'method':METHOD,'conditioning':MODE,'step':step,'train_signature':signature(c),'model_id':c['model_id'],'revision':c['revision'],'rank':c['rank'],'lora_alpha':c['lora_alpha'],'adapter_sha256':sha256(p/'adapter.safetensors'),'state_sha256':sha256(p/'training.pt'),'best_validation_flow':best,'config':c}
    write_json(p/'metadata.json',metadata);write_json(root/'latest.json',{'checkpoint':str(p)})
    return p

def restore(c,model,opt,scaler,sampler,generator):
    root=Path(c['run_dir']);pointer=root/'latest.json'
    if not pointer.exists():
        if list((root/'checkpoints').glob('*/metadata.json')):raise ValueError('Checkpoint exists without latest pointer; inspect before resuming.')
        return 0,None
    p=Path(read_json(pointer)['checkpoint']);meta=read_json(p/'metadata.json')
    if meta.get('method')!=METHOD or meta['train_signature']!=signature(c):raise ValueError('Resume config/method mismatch; use a new run folder.')
    if sha256(p/'adapter.safetensors')!=meta['adapter_sha256'] or sha256(p/'training.pt')!=meta['state_sha256']:raise ValueError('Damaged checkpoint')
    load_adapter(model,load_file(str(p/'adapter.safetensors')))
    state=torch.load(p/'training.pt',map_location='cpu',weights_only=True)
    opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);sampler.setstate(state['sampler_state']);generator.set_state(state['noise_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    print('Resumed committed step',state['step'],flush=True)
    return state['step'],state['best_val']
