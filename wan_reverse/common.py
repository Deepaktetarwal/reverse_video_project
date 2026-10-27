import hashlib
import json
import os
from pathlib import Path

LATENT_KEYS = ['model_id','revision','height','width','frames']
TRAIN_KEYS = ['conditioning','precision','rank','lora_alpha','learning_rate','weight_decay','accumulate','gradient_clip','conditioning_dropout','sigma_min','sigma_max','flow_weight','kd_weight','temporal_weight','seed']

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def write_rows(path,values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in values),encoding='utf-8')
    os.replace(tmp,path)

def config(path):
    c=read_json(path)
    from .conditioning import MODE
    c.setdefault('conditioning',MODE)
    if c['conditioning']!=MODE: raise ValueError('This version requires original-prompt conditioning.')
    if c['height']%16 or c['width']%16 or (c['frames']-1)%4 or c['frames']<5:
        raise ValueError('Wan requires height/width multiples of 16 and frames=4k+1, at least 5 here.')
    if c['precision'] not in ('fp16','fp32','bf16'): raise ValueError('Bad precision')
    if not 0<c['sigma_min']<c['sigma_max']<1: raise ValueError('Bad sigma bounds')
    if c['accumulate']<1 or c['rank']<1: raise ValueError('Bad training configuration')
    for key in ('work_dir','run_dir'):
        Path(c[key]).mkdir(parents=True,exist_ok=True)
    return c

def safe_asset(root,relative):
    root=Path(root).resolve(); p=(root/relative).resolve()
    if not p.is_relative_to(root): raise ValueError('Asset escapes dataset directory')
    return p

def manifest(c):
    root=Path(c['dataset_dir'])
    p=root/('manifest_wan.jsonl' if c['accept_all_user_declared'] else 'accepted_manifest_wan.jsonl')
    data=rows(p)
    if not data: raise ValueError('Empty manifest. Review clips or explicitly use user-declared acceptance.')
    if len({r['id'] for r in data})!=len(data): raise ValueError('Duplicate IDs')
    if c.get('expected_candidates') is not None and len(data)!=c['expected_candidates']:
        raise ValueError(f"Expected {c['expected_candidates']} clips, found {len(data)}")
    groups={}
    for r in data:
        if r['model']!=c['model_id'] or r['revision']!=c['revision']: raise ValueError('Teacher revision mismatch')
        if r['split'] not in ('train','val','test'): raise ValueError('Bad split')
        group=r.get('prompt_id',r['id'])
        if group in groups and groups[group]!=r['split']: raise ValueError('Prompt group leaks across splits')
        groups[group]=r['split']
    return data

def prep_signature(c,data):
    return fingerprint({'settings':{k:c[k] for k in LATENT_KEYS},'records':data,'preprocess':'uniform-endpoints-resize-bilinear-frame-reverse-before-causal-vae-mode-v1'})

def prep_dir(c):
    return Path(c.get('prepared_run_dir',c['run_dir']))/'prepared'

def prep_index(c):
    data=manifest(c); signature=prep_signature(c,data)
    meta=read_json(prep_dir(c)/'metadata.json')
    if meta['signature']!=signature: raise ValueError('Prepared data/config mismatch. Use a new run_dir.')
    return meta,rows(prep_dir(c)/'index.jsonl')

def cache_dir(c):
    return Path(c.get('cache_run_dir',c['run_dir']))/'kd_cache'

def cache_signature(c,prepared):
    return fingerprint({'prepared':prepared,'draws':[c['cache_draws_train'],c['cache_draws_eval']], 'seed':c['seed'],'dropout':c['conditioning_dropout'],'sigma':[c['sigma_min'],c['sigma_max']], 'precision':c['precision'],'method':'paired-translation-flow-v1'})

def cache_index(c):
    prep,_=prep_index(c)
    meta=read_json(cache_dir(c)/'metadata.json')
    if meta['signature']!=cache_signature(c,prep['signature']): raise ValueError('KD cache/config mismatch')
    return meta,rows(cache_dir(c)/'index.jsonl')

def train_signature(c):
    meta,_=cache_index(c)
    return fingerprint({'cache':meta['signature'],'train':{k:c[k] for k in TRAIN_KEYS}})

def save_tensors(path,values,metadata=None):
    from safetensors.torch import save_file
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    save_file({k:v.detach().cpu().contiguous() for k,v in values.items()},str(tmp),metadata=metadata)
    os.replace(tmp,path)

def require_cuda():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('This stage requires a CUDA GPU. Use T4 in Colab.')

def dtype(c):
    import torch
    return {'fp16':torch.float16,'bf16':torch.bfloat16,'fp32':torch.float32}[c['precision']]

def arguments(description):
    import argparse
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--config',required=True)
    return p

_STAGED_ROOTS={}

def staged_dir(c,kind):
    key=fingerprint({'run_dir':c['run_dir'],'work_dir':c['work_dir'],'settings':{k:c[k] for k in LATENT_KEYS}})
    if key not in _STAGED_ROOTS:
        meta,_=prep_index(c)
        _STAGED_ROOTS[key]=Path(c['work_dir'])/'staged'/meta['signature']
    return _STAGED_ROOTS[key]/kind

def stage_training_files(c,include_cache=True):
    """Copy immutable tensors to local SSD once; checkpoints and original data stay on Drive."""
    import shutil
    _,prepared=prep_index(c)
    groups=[('prepared',prep_dir(c),prepared)]
    if include_cache:
        _,cached=cache_index(c); groups.append(('kd_cache',cache_dir(c),cached))
    for kind,source,entries in groups:
        target=staged_dir(c,kind); target.mkdir(parents=True,exist_ok=True)
        for row in entries:
            src=source/row['file']; dest=target/row['file']
            expected=read_json(src.with_suffix('.json'))['sha256']
            if dest.exists() and sha256(dest)==expected: continue
            if sha256(src)!=expected: raise ValueError('Cannot stage a damaged tensor file')
            tmp=dest.with_name(dest.name+'.tmp'); shutil.copy2(src,tmp)
            if sha256(tmp)!=expected: raise ValueError('Local tensor copy failed checksum')
            tmp.replace(dest)
        print('Local tensors ready:',kind,len(entries),flush=True)

def tensor_path(c,kind,filename):
    path=staged_dir(c,kind)/filename
    return path if path.exists() else (prep_dir(c) if kind=='prepared' else cache_dir(c))/filename
