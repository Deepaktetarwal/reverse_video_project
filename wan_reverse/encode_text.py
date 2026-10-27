"""Optional new UI prompt encoding. No training gradients; process exits before sampling."""
from pathlib import Path
import torch
from .common import *
from .conditioning import MODE

def main():
    p=arguments('Encode an original forward-action prompt for the reverse-time student.')
    p.add_argument('--prompt',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); c=config(a.config); require_cuda(); out=Path(a.out)
    signature=fingerprint({'model':c['model_id'],'revision':c['revision'],'prompt':a.prompt,'negative':'','encoder':'nf4-fp32-512','conditioning':MODE})
    marker=out.with_suffix('.json')
    if marker.exists() and out.exists():
        m=read_json(marker)
        if m['signature']==signature and sha256(out)==m['sha256']:
            print('Reusing encoded prompt.',flush=True); return
    from transformers import AutoTokenizer,UMT5EncoderModel,BitsAndBytesConfig
    from diffusers import WanPipeline
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)
    enc=UMT5EncoderModel.from_pretrained(c['model_id'],subfolder='text_encoder',revision=c['revision'],quantization_config=q,device_map={'':0},torch_dtype=torch.float32,low_cpu_mem_usage=True,use_safetensors=True).eval()
    tok=AutoTokenizer.from_pretrained(c['model_id'],subfolder='tokenizer',revision=c['revision'])
    pipe=WanPipeline(tokenizer=tok,text_encoder=enc,transformer=None,vae=None,scheduler=None)
    with torch.inference_mode():
        pos,neg=pipe.encode_prompt(a.prompt,negative_prompt='',max_sequence_length=512,device=torch.device('cuda'),dtype=torch.float32)
    if not torch.isfinite(pos).all() or not torch.isfinite(neg).all(): raise RuntimeError('Non-finite embeddings')
    save_tensors(out,{'forward_embeds':pos.float(),'negative_embeds':neg.float()})
    write_json(marker,{'signature':signature,'sha256':sha256(out),'prompt':a.prompt})
    print('Prompt encoded. Model released when this process exits.',flush=True)
if __name__=='__main__': main()
