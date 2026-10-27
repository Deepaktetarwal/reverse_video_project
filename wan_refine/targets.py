"""Inspect what the student is actually asked to reconstruct, before retraining."""
import argparse,json
from pathlib import Path
import torch
from safetensors.torch import load_file
from diffusers.utils import export_to_video
from wan_reverse.common import prep_index,prep_dir,write_json
from wan_reverse.models import vae,denormalize
from .core import load_config

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--id',required=True);p.add_argument('--per-id',action='store_true')
    a=p.parse_args();c=load_config(a.config);_,data=prep_index(c)
    row=next(r for r in data if r['id']==a.id)
    d=load_file(str(prep_dir(c)/row['file']));out=Path(c['run_dir'])/'target_check'
    if a.per_id:out=out/row['id']
    out.mkdir(parents=True,exist_ok=True)
    model=vae(c)
    for key,label in [('zf','forward_vae_reconstruction'),('zr','reverse_vae_reconstruction')]:
        with torch.inference_mode():x=model.decode(denormalize(d[key].to('cuda'),model),return_dict=False)[0]
        frames=((x[0].permute(1,2,3,0).float().clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
        export_to_video(list(frames),str(out/(label+'.mp4')),fps=c['fps'])
        del x,frames
    write_json(out/'target_summary.json',{'id':row['id'],'original_input_prompt':row['prompt'],'latent_shape':list(d['zr'].shape),'latent_pair_mse':float((d['zr']-d['zf']).square().mean()),'note':'Both latents decoded in stored order, without frame reversal. Visually verify reverse reconstruction before training.'})
    print((out/'target_summary.json').read_text())
if __name__=='__main__':main()
