import json,random,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from .common import *
from .models import transformer
from .adapters import trainable,load_adapter
from .training import CachedBatch,validation,checkpoint,resolve_checkpoint,verify_checkpoint

def main():
    p=arguments('Train original-prompt reverse-video Wan LoRA with flow matching and offline paired teacher KD.')
    p.add_argument('--resume',choices=['auto','none'],default='auto')
    a=p.parse_args(); c=config(a.config); require_cuda()
    meta,entries=cache_index(c)
    if not meta['complete']: raise ValueError('Finish full KD cache before training')
    smoke=read_json(Path(c['run_dir'])/'gpu_smoke.json')
    if smoke.get('train_signature')!=train_signature(c): raise ValueError('Rerun GPU smoke for this configuration')
    if smoke['adapter_max_change']<=0: raise ValueError('Pass GPU smoke first')
    train=[r for r in entries if r['split']=='train']; val=[r for r in entries if r['split']=='val']
    if not train or not val: raise ValueError('Missing train/val entries')
    stage_training_files(c)
    torch.manual_seed(c['seed']); random.seed(c['seed'])
    model,names=transformer(c,student=True,checkpointing=True)
    params=trainable(model)
    opt=torch.optim.AdamW(params,lr=c['learning_rate'],weight_decay=c['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=c['precision']=='fp16',init_scale=1024)
    sampler=random.Random(c['seed']); start_step=0; best=float('inf')
    latest=Path(c['run_dir'])/'latest.json'
    if a.resume=='auto' and latest.exists():
        path=resolve_checkpoint(c,'latest'); verify_checkpoint(c,path,training=True)
        load_adapter(model,load_file(str(path/'adapter.safetensors')))
        state=torch.load(path/'training.pt',map_location='cpu',weights_only=True)
        opt.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        sampler.setstate(state['sampler_state']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_step=state['step']; best=state['best_val']
        print('Resumed committed step',start_step,flush=True)
    elif latest.exists(): raise ValueError('Existing run found. Resume it or choose a new run_dir.')
    elif list((Path(c['run_dir'])/'checkpoints').glob('*/metadata.json')): raise ValueError('Committed checkpoints exist but latest.json is absent; restore the pointer or use a new run_dir.')
    if start_step>=c['max_steps']:
        print('Requested optimizer steps already completed.',flush=True); return
    opt.zero_grad(set_to_none=True)
    logfile=Path(c['run_dir'])/'training_log.jsonl'
    for step in range(start_step+1,c['max_steps']+1):
        begin=time.perf_counter(); summed={'flow':0.,'kd':0.,'temporal':0.}
        for _ in range(c['accumulate']):
            batch=CachedBatch(c,sampler.choice(train)); loss,parts=batch.loss(model)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss; stop and inspect precision')
            scaler.scale(loss/c['accumulate']).backward()
            for k,v in parts.items(): summed[k]+=float(v)/c['accumulate']
            del batch,loss
        scaler.unscale_(opt)
        grad=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip'],error_if_nonfinite=True)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        row={'step':step,**summed,'gradient_norm':float(grad),'seconds':time.perf_counter()-begin}
        val_result=None
        if step%c['validate_every']==0 or step==c['max_steps']:
            val_result=validation(model,c,val)
            row.update({'val_'+k:v for k,v in val_result.items()})
        is_best=val_result is not None and val_result['flow']<best
        if is_best: best=val_result['flow']
        # Store logs before checkpoint: repeats after a crash can be identified by step.
        with logfile.open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
        if step%c['checkpoint_every']==0 or step==c['max_steps'] or is_best:
            path=checkpoint(c['run_dir'],model,opt,scaler,c,step,best,sampler)
            if is_best: write_json(Path(c['run_dir'])/'best.json',{'checkpoint':str(path),'validation_flow':best})
    write_json(Path(c['run_dir'])/'training_complete.json',{'steps':c['max_steps'],'train_signature':train_signature(c),'best_validation_flow':best,'adapted_modules':names})
    print('Training complete. Generate held-out comparisons and evaluate before claiming success.',flush=True)
if __name__=='__main__': main()
