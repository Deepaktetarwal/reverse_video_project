import argparse,json,random,time,tempfile
from pathlib import Path
import torch
from safetensors.torch import load_file
from wan_reverse.common import write_json,read_json,stage_training_files,cache_index,tensor_path
from wan_reverse.models import transformer,predict
from wan_reverse.adapters import trainable,adapter_state,load_adapter
from wan_reverse.conditioning import student_embedding
from .core import *

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--until',type=int);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();c=load_config(a.config,allow_partial=a.smoke);root=Path(c['run_dir'])
    data=get_data(c);train=[r for r in data if r['id'] in c['train_ids']]
    val=train if c['phase']=='overfit' else [r for r in data if r['split']=='val']
    stage_training_files(c,include_cache=c['kd_weight']>0)
    torch.manual_seed(c['seed']);sampler=random.Random(c['seed']);generator=torch.Generator('cpu').manual_seed(c['seed']+999)
    model,names=transformer(c,student=True,checkpointing=True);params=trainable(model)
    opt=torch.optim.AdamW(params,lr=c['learning_rate'],weight_decay=c['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=c['precision']=='fp16',init_scale=1024)
    if a.smoke:
        d=read_clip(c,train[0]);sigma,noise,emb=draw(d,c,generator)
        before=adapter_state(model);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
        loss,parts=loss_on(model,d,sigma,noise,emb,c);scaler.scale(loss).backward();scaler.unscale_(opt)
        grad=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip'],error_if_nonfinite=True)
        scaler.step(opt);scaler.update();torch.cuda.synchronize();seconds=time.perf_counter()-begin
        updated=adapter_state(model);change=max(float((updated[k]-before[k]).abs().max()) for k in updated)
        model.eval();xr=(1-sigma)*d['zr'].float()+sigma*noise
        with torch.no_grad():first=predict(model,xr,sigma,emb,c)
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'adapter.safetensors';save_tensors(file,updated)
            load_adapter(model,{k:torch.zeros_like(v) for k,v in updated.items()});load_adapter(model,load_file(str(file)))
            with torch.no_grad():second=predict(model,xr,sigma,emb,c)
        error=float((first-second).abs().max())
        if change<=0 or not torch.isfinite(loss) or not torch.allclose(first,second,atol=1e-4,rtol=1e-4):raise RuntimeError('GPU smoke failed')
        result={'signature':signature(c),'gpu':torch.cuda.get_device_name(0),'loss':float(loss),'gradient_norm':float(grad),'adapter_max_change':change,'reload_max_abs_error':error,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'microbatch_update_seconds':seconds,'frames':c['frames'],'height':c['height'],'width':c['width'],'latent_shape':list(d['zr'].shape),'rough_train_compute_hours_excluding_validation':seconds*c['accumulate']*c['max_steps']/3600,'trainable_parameters':sum(x.numel() for x in params),'adapted_modules':len(names),'note':'Execution check, not reverse-motion learning evidence.'}
        write_json(root/'gpu_smoke_v3.json',result);print(json.dumps(result,indent=2));return
    if read_json(root/'gpu_smoke_v3.json')['signature']!=signature(c):raise ValueError('Run V3 GPU smoke for this config first.')
    step,best=restore(c,model,opt,scaler,sampler,generator)
    until=a.until or c['max_steps']
    if until>c['max_steps'] or until<1:raise ValueError('Invalid --until')
    if step>=until:print('Requested steps already committed.');return
    if best is None:
        baseline=validation(model,c,val,split='train' if c['phase']=='overfit' else 'val',compare_base=True)
        best=baseline['flow'];write_json(root/'baseline_validation.json',baseline);print('Baseline:',json.dumps(baseline),flush=True)
    cached=[]
    if c['kd_weight']:
        _,cached=cache_index(c);cached=[r for r in cached if r['id'] in c['train_ids'] and r['split']=='train']
    opt.zero_grad(set_to_none=True);skipped=0
    while step<until:
        begin=time.perf_counter();values=[];kd_value=0.;proposed_step=step+1
        for micro in range(c['accumulate']):
            row=sampler.choice(train);d=read_clip(c,row);sigma,noise,emb=draw(d,c,generator)
            loss,parts=loss_on(model,d,sigma,noise,emb,c)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite objective')
            scaler.scale(loss/c['accumulate']).backward()
            values.append({k:float(v.detach()) for k,v in parts.items()})
            del d,loss,parts
        # Optional OLD transported teacher residual, isolated as an explicit ablation.
        if cached and proposed_step%c['kd_every']==0:
            spec=sampler.choice(cached)
            d=load_file(str(tensor_path(c,'prepared',spec['prepared'])));k=load_file(str(tensor_path(c,'kd_cache',spec['file'])))
            zr=d['zr'].to('cuda');eps=k['noise'].to('cuda');s=float(k['sigma'])
            xr=(1-s)*zr+s*eps;emb=student_embedding(d,bool(k['drop'])).to('cuda')
            kd_loss=(predict(model,xr,s,emb,c)-k['kd'].to('cuda')).square().mean()
            # Average KD contribution is kd_weight over all optimizer updates.
            scaler.scale(c['kd_weight']*c['kd_every']*kd_loss).backward();kd_value=float(kd_loss.detach())
            del d,k,kd_loss
        scaler.unscale_(opt);grad=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip'],error_if_nonfinite=True)
        scale=scaler.get_scale();scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        if scaler.get_scale()<scale:
            skipped+=1;print('FP16 overflow: optimizer update skipped, reducing loss scale.',flush=True)
            if skipped>20:raise RuntimeError('Too many FP16 overflows; inspect the loss/precision.')
            continue
        skipped=0;step=proposed_step
        record={'step':step,**{k:sum(r[k] for r in values)/len(values) for k in values[0]},'kd':kd_value,'gradient_norm':float(grad),'seconds':time.perf_counter()-begin}
        is_best=False
        if step%c['validate_every']==0 or step==until:
            result=validation(model,c,val,split='train' if c['phase']=='overfit' else 'val',compare_base=True)
            record['validation']=result
            is_best=result['flow']<best
            if is_best:best=result['flow']
        with (root/'training_log_v3.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
        if step%c['checkpoint_every']==0 or step==until or is_best:
            path=save_checkpoint(c,model,opt,scaler,step,best,sampler,generator)
            if is_best:write_json(root/'best.json',{'checkpoint':str(path),'validation_flow':best})
    if step==c['max_steps']:write_json(root/'training_complete.json',{'steps':step,'signature':signature(c),'phase':c['phase'],'note':'Evaluate generated motion; optimization completion is not task success.'})
if __name__=='__main__':main()
