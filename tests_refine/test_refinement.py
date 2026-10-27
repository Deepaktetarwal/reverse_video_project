import json,random
from pathlib import Path
import torch
import pytest
from wan_refine.objective import objective
from wan_refine.core import draw,save_checkpoint,restore
from wan_reverse.adapters import inject,trainable,adapter_state

torch.set_num_threads(1)

def weights():
    return dict(order_floor=.01,order_sigma_min=.2,order_sigma_max=.85,min_pair_separation=1e-4,order_margin=.2,motion_floor=.05,motion_weight=.1,order_weight=.1,sigma_min=.02,sigma_max=.98,conditioning_dropout=0.)

def test_exact_reverse_velocity_satisfies_every_objective():
    zr,zf,eps=[torch.randn(1,4,5,4,4) for _ in range(3)]
    loss,parts=objective(eps-zr,zr,zf,eps,.5,weights())
    assert float(loss)<1e-10
    assert float(parts['order_gap'])==pytest.approx(1,abs=1e-5)

def test_forward_reconstruction_is_penalized_and_gradient_moves_toward_reverse():
    zr=torch.arange(5.).view(1,1,5,1,1).expand(1,2,5,2,2)
    zf=zr.flip(2);eps=torch.randn_like(zr);s=.5
    xr=(1-s)*zr+s*eps
    pred=((xr-zf)/s).detach().requires_grad_()
    loss,p=objective(pred,zr,zf,eps,s,weights())
    assert float(p['order_gap'])==pytest.approx(-1)
    assert float(p['order'])==pytest.approx(1.2)
    grad=torch.autograd.grad(loss,pred)[0]
    # Gradient descent must decrease squared error to the true reverse velocity.
    assert float((grad*(pred.detach()-(eps-zr))).sum())>0
    corrected=pred.detach()-.01*grad
    new,_=objective(corrected,zr,zf,eps,s,weights())
    assert float(new)<float(loss)

def test_small_separation_and_static_targets_do_not_create_false_margin():
    zr=torch.randn(1,2,5,2,2);zf=zr+.02;eps=torch.randn_like(zr)
    loss,p=objective(eps-zr,zr,zf,eps,.5,weights())
    assert float(p['order'])<1e-7
    z=torch.zeros_like(zr);pred=torch.ones_like(zr,requires_grad=True)
    loss,p=objective(pred,z,z,eps,.5,weights());loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert float(p['order_active'])==0

def test_fresh_noise_changes_between_draws_and_rng_state_replays():
    c=weights();d={'zr':torch.zeros(1,2,5,2,2),'forward_embeds':torch.ones(1,4,8),'negative_embeds':torch.zeros(1,4,8)}
    g=torch.Generator().manual_seed(17)
    first=draw(d,c,g);state=g.get_state();second=draw(d,c,g)
    assert not torch.equal(first[1],second[1])
    g.set_state(state);again=draw(d,c,g)
    assert second[0]==again[0] and torch.equal(second[1],again[1])
    assert torch.equal(first[2],d['forward_embeds'])

def tiny_wan():
    from diffusers import WanTransformer3DModel
    return WanTransformer3DModel(patch_size=(1,2,2),num_attention_heads=2,attention_head_dim=16,in_channels=4,out_channels=4,text_dim=8,freq_dim=16,ffn_dim=64,num_layers=1,rope_max_seq_len=32)

def test_rank16_real_wan_checkpointed_backward_and_frozen_base():
    torch.manual_seed(3);m=tiny_wan();inject(m,16,16);m.enable_gradient_checkpointing();m.train()
    base={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
    zr,zf,eps=[torch.randn(1,4,3,4,4) for _ in range(3)]
    s=.5;x=(1-s)*zr+s*eps;text=torch.randn(1,5,8);t=torch.tensor([500.])
    optimizer=torch.optim.AdamW(trainable(m),lr=.003)
    first=None
    for i in range(8):
        pred=m(x,t,text,return_dict=False)[0]
        loss,_=objective(pred,zr,zf,eps,s,weights())
        if first is None:first=float(loss.detach())
        optimizer.zero_grad();loss.backward()
        assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
        optimizer.step()
    with torch.no_grad():last,_=objective(m(x,t,text,return_dict=False)[0],zr,zf,eps,s,weights())
    assert float(last)<first
    for n,p in m.named_parameters():
        if n in base:assert torch.equal(p,base[n])

def test_v3_checkpoint_restores_optimizer_and_fresh_noise_rng(tmp_path,monkeypatch):
    import wan_refine.core as core
    monkeypatch.setattr(core,'signature',lambda c:'fixture')
    m=tiny_wan();inject(m,16,16)
    opt=torch.optim.AdamW(trainable(m),lr=.001);scaler=torch.amp.GradScaler('cuda',enabled=False)
    sampler=random.Random(13);g=torch.Generator().manual_seed(14)
    c={'run_dir':str(tmp_path),'model_id':'fixture','revision':'r','rank':16,'lora_alpha':16}
    save_checkpoint(c,m,opt,scaler,25,.4,sampler,g)
    expected=torch.randn(10,generator=g);expected_sample=sampler.random()
    with torch.no_grad():
        for p in trainable(m):p.add_(7)
    step,best=restore(c,m,opt,scaler,sampler,g)
    assert step==25 and best==.4
    assert torch.equal(torch.randn(10,generator=g),expected)
    assert sampler.random()==expected_sample
    meta=json.loads((tmp_path/'checkpoints/step_000025/metadata.json').read_text())
    assert meta['method']==core.METHOD
    monkeypatch.setattr(core,'signature',lambda c:'different')
    with pytest.raises(ValueError,match='mismatch'):restore(c,m,opt,scaler,sampler,g)

def test_shared_preparation_does_not_redirect_new_run_checkpoints(tmp_path):
    from wan_reverse.common import prep_dir,cache_dir
    c={'run_dir':str(tmp_path/'v3'),'prepared_run_dir':str(tmp_path/'v2'),'cache_run_dir':str(tmp_path/'v2')}
    assert prep_dir(c)==tmp_path/'v2/prepared'
    assert cache_dir(c)==tmp_path/'v2/kd_cache'
    assert Path(c['run_dir'])==tmp_path/'v3'

def test_actual_train_entrypoint_commits_and_resumes_updates_on_cpu(tmp_path,monkeypatch):
    import sys
    import wan_refine.core as core
    import wan_refine.train as entry
    c=dict(weights(),run_dir=str(tmp_path),precision='fp32',seed=23,phase='overfit',train_ids=['x'],rank=16,lora_alpha=16,learning_rate=.003,weight_decay=.01,gradient_clip=1.,accumulate=1,max_steps=3,kd_weight=0.,kd_every=4,validate_every=1,checkpoint_every=1,validation_draws=2,validation_seed=37,model_id='fixture',revision='r')
    row={'id':'x','split':'train'}
    d={'zr':torch.randn(1,4,3,4,4),'zf':torch.randn(1,4,3,4,4),'forward_embeds':torch.randn(1,5,8),'negative_embeds':torch.zeros(1,5,8)}
    def make_model(*args,**kwargs):
        torch.manual_seed(5);m=tiny_wan();names=inject(m,16,16);m.enable_gradient_checkpointing();m.train();return m,names
    def cpu_predict(model,x,sigma,embedding,config):
        return model(x,torch.tensor([1000*sigma]),embedding,return_dict=False)[0].float()
    monkeypatch.setattr(entry,'load_config',lambda path,**kwargs:c)
    monkeypatch.setattr(entry,'get_data',lambda config:[row])
    monkeypatch.setattr(entry,'stage_training_files',lambda *args,**kwargs:None)
    monkeypatch.setattr(entry,'transformer',make_model)
    monkeypatch.setattr(entry,'read_clip',lambda *args,**kwargs:d)
    monkeypatch.setattr(core,'read_clip',lambda *args,**kwargs:d)
    monkeypatch.setattr(core,'predict',cpu_predict)
    monkeypatch.setattr(entry,'signature',lambda config:'fixture')
    monkeypatch.setattr(core,'signature',lambda config:'fixture')
    (tmp_path/'gpu_smoke_v3.json').write_text(json.dumps({'signature':'fixture'}))
    monkeypatch.setattr(sys,'argv',['train','--config','fixture','--until','2']);entry.main()
    state=torch.load(tmp_path/'checkpoints/step_000002/training.pt',weights_only=True)
    assert state['optimizer']['state']
    monkeypatch.setattr(sys,'argv',['train','--config','fixture','--until','3']);entry.main()
    log=[json.loads(s) for s in (tmp_path/'training_log_v3.jsonl').read_text().splitlines()]
    assert [r['step'] for r in log]==[1,2,3]
    assert json.loads((tmp_path/'training_complete.json').read_text())['steps']==3
    assert json.loads((tmp_path/'checkpoints/step_000003/metadata.json').read_text())['method']==core.METHOD
