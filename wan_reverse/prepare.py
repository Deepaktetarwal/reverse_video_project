import os
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from .common import *
from .models import vae,normalize

def preprocess(frames,c):
    if c.get('require_all_81_frames') and (len(frames)!=81 or c['frames']!=81):raise ValueError('This experiment requires exactly all 81 source frames; no repetition or temporal subsampling is allowed.')
    # Cover the entire original action, preserving endpoints; reduce for T4 backward memory.
    indices=np.rint(np.linspace(0,len(frames)-1,c['frames'])).astype(np.int64)
    x=torch.from_numpy(frames[indices].copy()).permute(0,3,1,2).float()/255
    x=F.interpolate(x,size=(c['height'],c['width']),mode='bilinear',align_corners=False)
    return x.permute(1,0,2,3).unsqueeze(0).contiguous(),indices

def main():
    p=arguments('Encode forward and reversed pixels separately with Wan temporal VAE.')
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args(); c=config(a.config); require_cuda()
    data=manifest(c); signature=prep_signature(c,data); out=prep_dir(c); out.mkdir(exist_ok=True)
    meta=out/'metadata.json'
    if meta.exists() and read_json(meta)['signature']!=signature: raise ValueError('Use a new run_dir for changed preprocessing/data.')
    index=[]; needed=[]
    for r in data:
        file=out/f"{r['id']}.safetensors"
        marker=file.with_suffix('.json')
        if marker.exists():
            info=read_json(marker)
            if info['signature']!=signature or not file.exists() or sha256(file)!=info['sha256']: raise ValueError('Corrupt/mismatched prepared file')
            index.append(info['row'])
        else: needed.append(r)
    if a.limit>0: needed=needed[:a.limit]
    if needed:
        v=vae(c)
        for i,r in enumerate(needed):
            asset=r['assets']['frames_uint8']; source=safe_asset(c['dataset_dir'],asset['path'])
            if sha256(source)!=asset['sha256']: raise ValueError('Frame checksum mismatch')
            with np.load(source,allow_pickle=False) as f: x,indices=preprocess(f['frames'],c)
            x=x.to('cuda')*2-1
            with torch.inference_mode():
                zf=normalize(v.encode(x).latent_dist.mode(),v)
                # Causal temporal VAE: reverse PIXELS before encoding.
                zr=normalize(v.encode(x.flip(2).contiguous()).latent_dist.mode(),v)
            easset=r['assets']['embedding']; nasset=r['assets']['negative_embedding']
            ep=safe_asset(c['dataset_dir'],easset['path']); npth=safe_asset(c['dataset_dir'],nasset['path'])
            if sha256(ep)!=easset['sha256'] or sha256(npth)!=nasset['sha256']: raise ValueError('Embedding checksum mismatch')
            emb=load_file(str(ep)); neg=load_file(str(npth))
            # Retain the legacy reverse embedding only for cache compatibility; v2 student uses forward_embeds.
            values={'zf':zf.float(),'zr':zr.float(),'forward_embeds':emb['forward_prompt_embeds'].float(),'reverse_embeds':emb['reverse_prompt_embeds'].float(),'negative_embeds':neg['negative_prompt_embeds'].float()}
            if not all(torch.isfinite(t).all() for t in values.values()): raise RuntimeError('Non-finite prepared tensors')
            dest=out/f"{r['id']}.safetensors"; save_tensors(dest,values)
            row={k:r[k] for k in ('id','event','split','prompt','reverse_caption')}
            row.update(file=dest.name,indices=indices.tolist(),latent_shape=list(zf.shape))
            write_json(dest.with_suffix('.json'),{'signature':signature,'sha256':sha256(dest),'row':row})
            index.append(row)
            print(f"Prepared {r['id']}: {list(zf.shape)}",flush=True)
            del x,zf,zr,values
    order={r['id']:i for i,r in enumerate(data)}; index.sort(key=lambda r:order[r['id']])
    write_rows(out/'index.jsonl',index)
    write_json(meta,{'signature':signature,'complete':len(index)==len(data),'clips':len(index),'expected':len(data),'preprocess':{k:c[k] for k in LATENT_KEYS}})
    print(f'Prepared {len(index)}/{len(data)} clips. Rerun to finish partial work.',flush=True)
if __name__=='__main__': main()
