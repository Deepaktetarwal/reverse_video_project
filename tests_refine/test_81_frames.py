import json
from pathlib import Path
import numpy as np
import pytest
import torch
from wan_reverse.prepare import preprocess
from wan_reverse.common import prep_signature
from wan_reverse.adapters import inject,trainable
from wan_refine.objective import objective

torch.set_num_threads(1)

def test_all_81_source_frames_are_retained_without_repetition():
    source=np.stack([np.full((8,8,3),i,dtype=np.uint8) for i in range(81)])
    c={'frames':81,'height':16,'width':32,'require_all_81_frames':True}
    x,indices=preprocess(source,c)
    assert indices.tolist()==list(range(81))
    assert x.shape==(1,3,81,16,32)
    assert torch.allclose(x[0,0,:,0,0],torch.arange(81.)/255,atol=1e-6)
    for n in (17,80,82):
        with pytest.raises(ValueError,match='exactly all 81'):preprocess(np.zeros((n,8,8,3),np.uint8),c)

def test_81_preparation_signature_rejects_17_frame_artifacts():
    c={'model_id':'m','revision':'r','height':256,'width':448,'frames':17}
    assert prep_signature(c,[])!=prep_signature(dict(c,frames=81),[])

def test_partial_preparation_is_smoke_only_and_checks_full_train_split(monkeypatch):
    import wan_refine.core as core
    c={'method':core.METHOD,'order_margin':.2,'kd_weight':0,'kd_every':4,'train_ids':['a','b']}
    monkeypatch.setattr(core,'config',lambda path:c)
    monkeypatch.setattr(core,'prep_index',lambda config:({'complete':False},[{'id':'a','split':'train'}]))
    monkeypatch.setattr(core,'manifest',lambda config:[{'id':'a','split':'train'},{'id':'b','split':'train'},{'id':'v','split':'val'}])
    assert core.load_config('fixture',allow_partial=True)==c
    with pytest.raises(ValueError,match='Partial preparation'):core.load_config('fixture')
    c['train_ids']=['v']
    with pytest.raises(ValueError,match='training clip'):core.load_config('fixture',allow_partial=True)

def test_real_tiny_wan_81_frame_latent_sampling_and_backward():
    from diffusers import WanPipeline,WanTransformer3DModel,UniPCMultistepScheduler
    m=WanTransformer3DModel(patch_size=(1,2,2),num_attention_heads=2,attention_head_dim=16,in_channels=4,out_channels=4,text_dim=8,freq_dim=16,ffn_dim=64,num_layers=1,rope_max_seq_len=32)
    inject(m,16,16)
    scheduler=UniPCMultistepScheduler(num_train_timesteps=1000,prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=8)
    pipe=WanPipeline(tokenizer=None,text_encoder=None,vae=None,scheduler=scheduler,transformer=m)
    emb=torch.randn(1,5,8)
    m.eval()
    with torch.no_grad():z=pipe(prompt_embeds=emb,negative_prompt_embeds=torch.zeros_like(emb),height=16,width=16,num_frames=81,num_inference_steps=3,guidance_scale=1.,generator=torch.Generator().manual_seed(81),output_type='latent').frames
    assert z.shape==(1,4,21,2,2) and torch.isfinite(z).all()
    m.train();m.enable_gradient_checkpointing()
    zr,zf,noise=[torch.randn_like(z) for _ in range(3)]
    xr=.5*zr+.5*noise
    pred=m(xr,torch.tensor([500.]),emb,return_dict=False)[0]
    c={'order_floor':.01,'order_sigma_min':.2,'order_sigma_max':.85,'min_pair_separation':1e-4,'order_margin':.2,'motion_floor':.05,'motion_weight':.1,'order_weight':.1}
    loss,_=objective(pred,zr,zf,noise,.5,c);loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in trainable(m))
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)

def test_real_small_vae_encodes_81_to_21_and_decodes_back_to_81():
    from diffusers import AutoencoderKLWan
    from wan_reverse.models import normalize,denormalize
    v=AutoencoderKLWan(base_dim=8,decoder_base_dim=8,z_dim=4,dim_mult=[1,2,2,2],num_res_blocks=1,latents_mean=[0.]*4,latents_std=[1.]*4).eval()
    pixels=torch.randn(1,3,81,16,16).clamp(-1,1)
    with torch.no_grad():
        z=normalize(v.encode(pixels).latent_dist.mode(),v)
        decoded=v.decode(denormalize(z,v),return_dict=False)[0]
    assert z.shape==(1,4,21,2,2)
    assert decoded.shape==pixels.shape and torch.isfinite(decoded).all()
