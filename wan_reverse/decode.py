from pathlib import Path
import time
import numpy as np
import torch
from safetensors.torch import load_file
from .common import *
from .models import vae,denormalize

def write_video(path,pixels,fps):
    from diffusers.utils import export_to_video
    # Diffusers expects float frames in [0,1]. Store the same uint8 values losslessly alongside MP4.
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    export_to_video(pixels.astype(np.float32)/255,str(path),fps=fps)
    np.savez_compressed(path.with_suffix('.npz'),frames=pixels)

def main():
    p=arguments('Decode generated latents in their original order, with no temporal reversal.')
    p.add_argument('--sample',required=True)
    a=p.parse_args(); c=config(a.config); require_cuda(); meta=read_json(a.sample)
    if sha256(meta['latent_path'])!=meta['latent_sha256']: raise ValueError('Latent checksum mismatch')
    path=Path(a.sample).with_suffix('.mp4')
    done=path.with_suffix('.decoded.json')
    if done.exists():
        d=read_json(done)
        if d['latent_sha256']==meta['latent_sha256'] and path.exists() and path.with_suffix('.npz').exists() and sha256(path)==d['video_sha256']:
            print('Already decoded:',path,flush=True); return
    v=vae(c); z=load_file(meta['latent_path'])['latent'].to('cuda',torch.float32)
    torch.cuda.synchronize(); start=time.perf_counter()
    with torch.inference_mode(): x=v.decode(denormalize(z,v),return_dict=False)[0]
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    if not torch.isfinite(x).all(): raise RuntimeError('Non-finite decoded video')
    pixels=np.rint(((x[0].permute(1,2,3,0).float().cpu().numpy()+1)/2).clip(0,1)*255).astype(np.uint8)
    if pixels.shape!=(c['frames'],c['height'],c['width'],3): raise ValueError('Decoded shape differs from configuration')
    write_video(path,pixels,c['fps'])
    write_json(done,{'video':str(path),'video_sha256':sha256(path),'latent_sha256':meta['latent_sha256'],'decode_seconds':elapsed,'total_compute_seconds':elapsed+meta['denoise_seconds'],'frame_count':len(pixels),'fps':c['fps'],'mode':meta['mode'],'id':meta['id'],'frame_order':'unchanged from sampled latent'})
    print('Video:',path,flush=True)
if __name__=='__main__': main()
