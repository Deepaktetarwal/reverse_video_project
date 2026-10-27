from pathlib import Path
import torch
from safetensors.torch import load_file
from .common import *
from .models import transformer,predict
from .losses import interpolate,transport_target

def main():
    p=arguments('Cache frozen teacher predictions for paired latent-transport distillation.')
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args(); c=config(a.config); require_cuda()
    prep,data=prep_index(c)
    if not prep['complete']: raise ValueError('Finish prepare stage first')
    signature=cache_signature(c,prep['signature']); out=cache_dir(c); out.mkdir(exist_ok=True)
    if (out/'metadata.json').exists() and read_json(out/'metadata.json')['signature']!=signature: raise ValueError('Use a new run_dir for changed cache settings')
    pending=[]; index=[]
    for i,row in enumerate(data):
        draws=c['cache_draws_train'] if row['split']=='train' else c['cache_draws_eval']
        for j in range(draws):
            name=f"{row['id']}_d{j:03d}"; path=out/(name+'.safetensors'); marker=out/(name+'.json')
            spec={'id':row['id'],'event':row['event'],'split':row['split'],'draw':j,'prepared':row['file'],'file':path.name,'seed':c['seed']+i*10000+j}
            if marker.exists():
                info=read_json(marker)
                if info['signature']!=signature or info['row']!=spec or not path.exists() or sha256(path)!=info['sha256']: raise ValueError('Corrupt/mismatched KD entry')
                index.append(spec)
            else: pending.append(spec)
    if a.limit>0: pending=pending[:a.limit]
    if pending:
        stage_training_files(c,include_cache=False)
        teacher,_=transformer(c)
        for spec in pending:
            d=load_file(str(tensor_path(c,'prepared',spec['prepared'])))
            zf,zr=d['zf'].to('cuda'),d['zr'].to('cuda')
            generator=torch.Generator('cpu').manual_seed(spec['seed'])
            sigma=float(torch.rand((),generator=generator))*(c['sigma_max']-c['sigma_min'])+c['sigma_min']
            noise=torch.randn(zr.shape,generator=generator,dtype=torch.float32).to('cuda')
            drop=spec['split']=='train' and float(torch.rand((),generator=generator))<c['conditioning_dropout']
            emb=d['negative_embeds' if drop else 'forward_embeds'].to('cuda')
            with torch.inference_mode():
                vf=predict(teacher,interpolate(zf,noise,sigma),sigma,emb,c)
                kd=transport_target(vf,zf,zr)
            if not torch.isfinite(kd).all(): raise RuntimeError('Non-finite frozen teacher target')
            path=out/spec['file']
            save_tensors(path,{'noise':noise.float(),'sigma':torch.tensor(sigma),'kd':kd.float(),'drop':torch.tensor(int(drop))})
            write_json(path.with_suffix('.json'),{'signature':signature,'sha256':sha256(path),'row':spec})
            index.append(spec)
            print(f"Cached {spec['id']} draw {spec['draw']} ({spec['split']})",flush=True)
    index.sort(key=lambda r:(r['id'],r['draw']))
    expected=sum(c['cache_draws_train'] if r['split']=='train' else c['cache_draws_eval'] for r in data)
    write_rows(out/'index.jsonl',index)
    write_json(out/'metadata.json',{'signature':signature,'prepared_signature':prep['signature'],'complete':len(index)==expected,'entries':len(index),'expected':expected,'method':'vf + zf - zr; shared-noise paired translation'})
    print(f'Cache {len(index)}/{expected}. Rerun to complete partial work.',flush=True)
if __name__=='__main__': main()
