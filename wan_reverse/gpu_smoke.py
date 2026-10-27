import tempfile,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from .common import *
from .models import transformer,predict
from .training import CachedBatch
from .adapters import trainable,adapter_state,load_adapter

def main():
    a=arguments('Real Wan LoRA backward/update/reload smoke test; no claim of learning.').parse_args()
    c=config(a.config); require_cuda(); _,entries=cache_index(c)
    spec=next(r for r in entries if r['split']=='train')
    stage_training_files(c)
    torch.manual_seed(c['seed'])
    model,names=transformer(c,student=True,checkpointing=True)
    params=trainable(model); opt=torch.optim.AdamW(params,lr=c['learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=c['precision']=='fp16',init_scale=1024)
    batch=CachedBatch(c,spec); before={k:v.clone() for k,v in adapter_state(model).items()}
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
    loss,_=batch.loss(model)
    if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    grad=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip'],error_if_nonfinite=True)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    updated=adapter_state(model)
    change=max(float((updated[k]-before[k]).abs().max()) for k in updated)
    if change==0: raise RuntimeError('LoRA parameters did not update')
    model.eval()
    with torch.no_grad(): first=predict(model,batch.xr,batch.sigma,batch.emb,c).cpu()
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'adapter.safetensors'; save_tensors(p,updated)
        load_adapter(model,{k:torch.zeros_like(v) for k,v in updated.items()})
        load_adapter(model,load_file(str(p)))
        with torch.no_grad(): second=predict(model,batch.xr,batch.sigma,batch.emb,c).cpu()
    error=float((first-second).abs().max())
    if not torch.allclose(first,second,atol=1e-4,rtol=1e-4): raise RuntimeError('Adapter roundtrip changed predictions')
    result={'train_signature':train_signature(c),'gpu':torch.cuda.get_device_name(0),'latent_shape':list(batch.zr.shape),'trainable_parameters':sum(p.numel() for p in params),'adapted_modules':len(names),'loss':float(loss),'gradient_norm':float(grad),'adapter_max_change':change,'reload_max_abs_error':error,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'forward_backward_update_seconds':elapsed,'rough_train_compute_hours_excluding_validation':elapsed*c['accumulate']*c['max_steps']/3600,'note':'One real update and serialization check only; not evidence of reverse-time learning.'}
    write_json(Path(c['run_dir'])/'gpu_smoke.json',result)
    print(__import__('json').dumps(result,indent=2))
if __name__=='__main__': main()
