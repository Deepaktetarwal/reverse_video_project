"""Original-prompt direct student or unadapted generation. No output frame reversal."""
import re,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from .common import *
from .models import transformer
from .adapters import load_adapter
from .training import resolve_checkpoint,verify_checkpoint
from .conditioning import MODE,student_embedding,student_prompt

def main():
    p=arguments(__doc__)
    p.add_argument('--mode',choices=['student','unadapted'],default='student')
    p.add_argument('--checkpoint',default='best'); p.add_argument('--id')
    p.add_argument('--embedding'); p.add_argument('--prompt',default='')
    p.add_argument('--seed',type=int,default=2026); p.add_argument('--out-dir',required=True)
    p.add_argument('--steps',type=int)
    p.add_argument('--adapter-scale',type=float,default=1.0)
    a=p.parse_args(); c=config(a.config); require_cuda()
    _,data=prep_index(c)
    if a.id:
        row=next(r for r in data if r['id']==a.id)
        embpath=prep_dir(c)/row['file']; prompt=student_prompt(row); key=row['id']
    elif a.embedding:
        embpath=Path(a.embedding); prompt=a.prompt; key='custom_'+fingerprint(prompt)[:12]
    else: p.error('Supply --id or --embedding with --prompt')
    em=load_file(str(embpath)); steps=a.steps or c['inference_steps']
    if steps<1: raise ValueError('Steps must be positive')
    ckpt=None; tag='base'
    if a.mode=='student':
        ckpt=resolve_checkpoint(c,a.checkpoint); info=verify_checkpoint(c,ckpt)
        if info['step']<1: raise ValueError('A trained adapter is required')
        tag=info['adapter_sha256'][:12]
    identity={'conditioning':MODE,'adapter_scale':a.adapter_scale,'prompt':prompt,'embedding_sha':sha256(embpath),'mode':a.mode,'adapter':tag,'seed':a.seed,'steps':steps,'height':c['height'],'width':c['width'],'frames':c['frames'],'guidance':c['guidance'],'flow_shift':c['flow_shift'],'model':c['model_id'],'revision':c['revision'],'precision':c['precision']}
    key=re.sub('[^A-Za-z0-9_-]','_',key)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    stem=f"{key}_{a.mode}_seed{a.seed}_{fingerprint(identity)[:10]}"
    latent=out/(stem+'.safetensors'); marker=out/(stem+'.json')
    if marker.exists():
        old=read_json(marker)
        if old['identity']!=identity or not latent.exists() or sha256(latent)!=old['latent_sha256']: raise ValueError('Mismatched/corrupt existing sample')
        write_json(out/'last_sample.json',{'metadata':str(marker)})
        print('Sample already exists:',marker,flush=True); return
    m,_=transformer(c,student=a.mode=='student')
    if ckpt: load_adapter(m,load_file(str(ckpt/'adapter.safetensors')))
    if a.adapter_scale<0 or a.adapter_scale>4: raise ValueError('Adapter scale must be between 0 and 4')
    from .adapters import LoRALinear
    for layer in m.modules():
        if isinstance(layer,LoRALinear): layer.scale*=a.adapter_scale
    m.eval()
    from diffusers import WanPipeline,UniPCMultistepScheduler
    scheduler=UniPCMultistepScheduler.from_pretrained(c['model_id'],subfolder='scheduler',revision=c['revision'],flow_shift=c['flow_shift'])
    pipe=WanPipeline(tokenizer=None,text_encoder=None,vae=None,scheduler=scheduler,transformer=m).to('cuda')
    def check(pipeline,i,t,kwargs):
        if not torch.isfinite(kwargs['latents']).all(): raise RuntimeError('Non-finite sampling state')
        return kwargs
    torch.manual_seed(a.seed); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=dtype(c),enabled=c['precision']!='fp32'):
        z=pipe(prompt_embeds=student_embedding(em).to('cuda',dtype(c)),negative_prompt_embeds=em['negative_embeds'].to('cuda',dtype(c)),height=c['height'],width=c['width'],num_frames=c['frames'],num_inference_steps=steps,guidance_scale=c['guidance'],generator=torch.Generator('cpu').manual_seed(a.seed),output_type='latent',callback_on_step_end=check).frames
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    save_tensors(latent,{'latent':z.float()})
    write_json(marker,{'identity':identity,'id':a.id or key,'prompt':prompt,'mode':a.mode,'latent_path':str(latent),'latent_sha256':sha256(latent),'denoise_seconds':elapsed,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'output_frame_order':'direct model output','checkpoint':str(ckpt) if ckpt else None})
    write_json(out/'last_sample.json',{'metadata':str(marker)})
    print('Latent ready:',marker,flush=True)
if __name__=='__main__': main()
