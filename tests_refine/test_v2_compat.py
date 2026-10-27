import json
from pathlib import Path
import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from diffusers import WanTransformer3DModel,WanPipeline,UniPCMultistepScheduler,AutoencoderKLWan
from wan_reverse.adapters import inject,trainable,adapter_state,load_adapter
from wan_reverse.common import save_tensors,safe_asset,manifest,prep_signature
from wan_reverse.losses import interpolate,transport_target,losses
from wan_reverse.models import normalize,denormalize
from wan_reverse.prepare import preprocess
from wan_reverse.evaluate import flow_metrics

torch.set_num_threads(1)

def tiny_wan():
    return WanTransformer3DModel(patch_size=(1,2,2),num_attention_heads=2,attention_head_dim=16,in_channels=4,out_channels=4,text_dim=8,freq_dim=16,ffn_dim=64,num_layers=1,rope_max_seq_len=32)

def test_paired_flow_transport_identity():
    zf,zr,eps=[torch.randn(1,4,3,4,4) for _ in range(3)]
    sigma=.63
    xf=interpolate(zf,eps,sigma); xr=interpolate(zr,eps,sigma)
    delta=zr-zf
    assert torch.allclose(xr,xf+(1-sigma)*delta,atol=1e-6)
    target=transport_target(eps-zf,zf,zr)
    assert torch.allclose(target,eps-zr,atol=1e-6)
    h=1e-3
    numerical=(interpolate(zr,eps,sigma+h)-interpolate(zr,eps,sigma-h))/(2*h)
    assert torch.allclose(target,numerical,atol=3e-4)

def test_temporal_auxiliary_is_zero_for_exact_flow():
    z=torch.randn(1,4,3,4,4); eps=torch.randn_like(z); sigma=.4
    target=eps-z; xr=interpolate(z,eps,sigma)
    loss,parts=losses(target,target,target,z,xr,sigma,dict(flow_weight=1,kd_weight=.1,temporal_weight=.05))
    assert loss.item()<1e-12

def test_real_wan_checkpointed_lora_backward_roundtrip(tmp_path):
    torch.manual_seed(7); model=tiny_wan()
    x=torch.randn(1,4,3,4,4); text=torch.randn(1,5,8); t=torch.tensor([500.])
    with torch.no_grad(): base=model(x,t,text,return_dict=False)[0]
    names=inject(model,2,2)
    assert len(names)==8
    assert all(p.dtype==torch.float32 for p in trainable(model))
    with torch.no_grad(): zero=model(x,t,text,return_dict=False)[0]
    assert torch.equal(base,zero)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    model.train();model.enable_gradient_checkpointing()
    optimizer=torch.optim.AdamW(trainable(model),lr=.01)
    pred=model(x,t,text,return_dict=False)[0]
    loss=(pred-torch.randn_like(pred)).square().mean();loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in trainable(model))
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step();optimizer.zero_grad()
    for n,p in model.named_parameters():
        if n in frozen: assert torch.equal(p,frozen[n])
    model.eval()
    with torch.no_grad(): updated=model(x,t,text,return_dict=False)[0]
    assert not torch.equal(updated,base)
    state=adapter_state(model); path=tmp_path/'adapter.safetensors';save_tensors(path,state)
    load_adapter(model,{k:torch.zeros_like(v) for k,v in state.items()})
    load_adapter(model,load_file(str(path)))
    with torch.no_grad(): reloaded=model(x,t,text,return_dict=False)[0]
    assert torch.equal(updated,reloaded)
    with pytest.raises(ValueError):load_adapter(model,{})

def test_real_tiny_wan_latent_pipeline():
    model=tiny_wan().eval();inject(model,2,2)
    scheduler=UniPCMultistepScheduler(num_train_timesteps=1000,prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=8)
    pipe=WanPipeline(tokenizer=None,text_encoder=None,vae=None,scheduler=scheduler,transformer=model)
    with torch.inference_mode():
        z=pipe(prompt_embeds=torch.randn(1,5,8),negative_prompt_embeds=torch.zeros(1,5,8),height=16,width=16,num_frames=9,num_inference_steps=3,guidance_scale=2,generator=torch.Generator().manual_seed(3),output_type='latent').frames
    assert z.shape==(1,4,3,2,2)
    assert torch.isfinite(z).all()

def test_small_real_wan_vae_normalization_and_reversal():
    torch.manual_seed(3)
    v=AutoencoderKLWan(base_dim=8,decoder_base_dim=8,z_dim=4,dim_mult=[1,2,2,2],num_res_blocks=1,latents_mean=[.1,.2,.3,.4],latents_std=[1.,2.,3.,4.]).eval()
    x=torch.randn(1,3,9,16,16).clamp(-1,1)
    with torch.inference_mode():
        zf=normalize(v.encode(x).latent_dist.mode(),v)
        zr=normalize(v.encode(x.flip(2)).latent_dist.mode(),v)
        y=v.decode(denormalize(zf,v),return_dict=False)[0]
    assert torch.allclose(denormalize(normalize(zf,v),v),zf,atol=1e-6)
    assert y.shape==x.shape
    assert torch.isfinite(y).all()
    assert not torch.allclose(zr,zf.flip(2),atol=1e-6)

def test_preprocess_preserves_endpoints_and_does_not_crop_action():
    frames=np.stack([np.full((20,30,3),i,dtype=np.uint8) for i in range(81)])
    x,indices=preprocess(frames,dict(frames=17,height=16,width=32))
    assert x.shape==(1,3,17,16,32)
    assert indices[0]==0 and indices[-1]==80
    assert float(x[0,0,0,0,0])==0
    assert abs(float(x[0,0,-1,0,0])-80/255)<1e-6

def test_manifest_split_leakage_and_path_guard(tmp_path):
    row=dict(id='x',prompt_id='same',split='train',model='m',revision='r')
    (tmp_path/'manifest_wan.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(dict(row,id='y',split='test'))+'\n')
    with pytest.raises(ValueError):manifest(dict(dataset_dir=str(tmp_path),accept_all_user_declared=True,model_id='m',revision='r'))
    with pytest.raises(ValueError):safe_asset(tmp_path,'../outside')

def test_flow_proxy_detects_motion_without_reward_claim():
    image=np.zeros((128,224,3),np.uint8);image[30:80,50:100]=255
    static=flow_metrics(np.stack([image,image,image]))
    assert static['warp_l1']==0
    assert static['flow_magnitude_pixels']<1e-5
    moving=flow_metrics(np.stack([image,np.roll(image,4,axis=1),np.roll(image,8,axis=1)]))
    assert moving['flow_magnitude_pixels']>static['flow_magnitude_pixels']+.1

def test_checkpoint_training_state_safe_reload(tmp_path,monkeypatch):
    import random
    import wan_reverse.training as training
    monkeypatch.setattr(training,'train_signature',lambda c:'fixture-signature')
    m=tiny_wan();inject(m,2,2)
    opt=torch.optim.AdamW(trainable(m),lr=.001)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    sampler=random.Random(12)
    c=dict(run_dir=str(tmp_path),model_id='fixture',revision='r',rank=2,lora_alpha=2)
    saved=training.checkpoint(tmp_path,m,opt,scaler,c,50,float('inf'),sampler)
    training.verify_checkpoint(c,saved,training=True)
    state=torch.load(saved/'training.pt',map_location='cpu',weights_only=True)
    restored=random.Random();restored.setstate(state['sampler_state'])
    assert restored.random()==sampler.random()
    assert state['step']==50
    assert json.loads((saved/'metadata.json').read_text())['best_validation_flow'] is None

def test_student_uses_original_prompt_and_embedding_only():
    from wan_reverse.conditioning import student_embedding,student_prompt
    saved={'forward_embeds':torch.tensor([1.]),'reverse_embeds':torch.tensor([-2.]),'negative_embeds':torch.tensor([0.])}
    assert torch.equal(student_embedding(saved),saved['forward_embeds'])
    assert torch.equal(student_embedding(saved,dropped=True),saved['negative_embeds'])
    record={'prompt':'A ball falls off a table.','reverse_caption':'A ball rises onto a table.'}
    assert student_prompt(record)=='A ball falls off a table.'

def test_conditioning_change_reuses_targets_but_invalidates_training_signature(monkeypatch):
    import wan_reverse.common as common
    c=json.loads((Path(__file__).resolve().parent/'fixtures/wan_t4.json').read_text())
    old=dict(c,conditioning='legacy_reverse_caption')
    assert common.prep_signature(c,[])==common.prep_signature(old,[])
    assert common.cache_signature(c,'prepared')==common.cache_signature(old,'prepared')
    monkeypatch.setattr(common,'cache_index',lambda config:({'signature':'same-immutable-cache'},[]))
    assert common.train_signature(c)!=common.train_signature(old)

def test_reverse_caption_checkpoint_is_rejected(tmp_path):
    from wan_reverse.training import verify_checkpoint
    (tmp_path/'metadata.json').write_text(json.dumps({'step':50}))
    with pytest.raises(ValueError,match='conditioning task'):
        verify_checkpoint({},tmp_path)

def test_migration_reuses_committed_caches_without_old_training_state(tmp_path,monkeypatch):
    import wan_reverse.migrate as migration
    from wan_reverse.common import sha256,write_json
    source=tmp_path/'old'; target=tmp_path/'new'
    monkeypatch.setattr(migration,'manifest',lambda c:[])
    monkeypatch.setattr(migration,'prep_signature',lambda c,data:'prep-signature')
    monkeypatch.setattr(migration,'cache_signature',lambda c,prep:'kd-signature')
    for kind,signature in [('prepared','prep-signature'),('kd_cache','kd-signature')]:
        directory=source/kind;directory.mkdir(parents=True)
        tensor=directory/'committed.safetensors'
        tensor.write_bytes(b'immutable tensor fixture')
        write_json(tensor.with_suffix('.json'),{'signature':signature,'sha256':sha256(tensor)})
        (directory/'interrupted.safetensors').write_bytes(b'no commit marker')
        write_json(directory/'metadata.json',{'signature':signature,'complete':False})
        (directory/'index.jsonl').write_text('')
    (source/'checkpoints').mkdir()
    (source/'checkpoints/old_adapter.safetensors').write_bytes(b'old task')
    write_json(source/'gpu_smoke.json',{'task':'old'})
    result=migration.migrate({'run_dir':str(target)},source)
    assert result['copied_tensors']=={'prepared':1,'kd_cache':1}
    assert (target/'prepared/committed.safetensors').read_bytes()==b'immutable tensor fixture'
    assert not (target/'prepared/interrupted.safetensors').exists()
    assert not (target/'checkpoints').exists()
    assert not (target/'gpu_smoke.json').exists()
    (target/'prepared/new_progress.txt').write_text('keep')
    assert migration.migrate({'run_dir':str(target)},source)==result
    assert (target/'prepared/new_progress.txt').read_text()=='keep'
