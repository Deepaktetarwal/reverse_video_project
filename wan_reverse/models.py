import torch
from .common import dtype,require_cuda

def transformer(c,student=False,checkpointing=False):
    from diffusers import WanTransformer3DModel
    require_cuda()
    model=WanTransformer3DModel.from_pretrained(c['model_id'],subfolder='transformer',revision=c['revision'],torch_dtype=dtype(c),low_cpu_mem_usage=True,use_safetensors=True).to('cuda')
    model.requires_grad_(False)
    names=[]
    if student:
        from .adapters import inject
        names=inject(model,c['rank'],c['lora_alpha'])
    if checkpointing: model.enable_gradient_checkpointing()
    model.train(student)
    return model,names

def vae(c):
    from diffusers import AutoencoderKLWan
    require_cuda()
    m=AutoencoderKLWan.from_pretrained(c['model_id'],subfolder='vae',revision=c['revision'],torch_dtype=torch.float32,low_cpu_mem_usage=True,use_safetensors=True).to('cuda').eval()
    m.requires_grad_(False); m.enable_tiling()
    return m

def normalize(z,v):
    mean=torch.tensor(v.config.latents_mean,device=z.device,dtype=z.dtype).view(1,v.config.z_dim,1,1,1)
    std=torch.tensor(v.config.latents_std,device=z.device,dtype=z.dtype).view(1,v.config.z_dim,1,1,1)
    return (z-mean)/std

def denormalize(z,v):
    mean=torch.tensor(v.config.latents_mean,device=z.device,dtype=z.dtype).view(1,v.config.z_dim,1,1,1)
    std=torch.tensor(v.config.latents_std,device=z.device,dtype=z.dtype).view(1,v.config.z_dim,1,1,1)
    return z*std+mean

def predict(model,x,sigma,embedding,c):
    t=torch.as_tensor([float(sigma)*1000],device=x.device,dtype=torch.float32)
    with torch.autocast('cuda',dtype=dtype(c),enabled=c['precision']!='fp32'):
        return model(hidden_states=x.to(dtype(c)),timestep=t,encoder_hidden_states=embedding.to(dtype(c)),return_dict=False)[0].float()
