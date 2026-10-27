"""Fixed-seed denoising diagnostics; these are NOT action-success measurements."""
import argparse,json
from pathlib import Path
from safetensors.torch import load_file
from wan_reverse.models import transformer
from wan_reverse.adapters import load_adapter
from wan_reverse.training import resolve_checkpoint,verify_checkpoint
from .core import *

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--split',choices=['train','val','test'],default='val');p.add_argument('--checkpoint',default='latest');p.add_argument('--unadapted',action='store_true')
    a=p.parse_args();c=load_config(a.config);stage_training_files(c,include_cache=False)
    rows=[r for r in get_data(c) if r['split']==a.split and (a.split!='train' or r['id'] in c['train_ids'])]
    model,_=transformer(c,student=not a.unadapted)
    if not a.unadapted:
        path=resolve_checkpoint(c,a.checkpoint);verify_checkpoint(c,path);load_adapter(model,load_file(str(path/'adapter.safetensors')))
    result=validation(model,c,rows,split=a.split,compare_base=not a.unadapted)
    result.update(split=a.split,mode='unadapted' if a.unadapted else 'student',checkpoint=a.checkpoint,note='Endpoint-order preference on noised real targets; not free-generation reverse-action success.')
    out=Path(c['run_dir'])/'diagnostics';write_json(out/(a.split+'_'+result['mode']+'.json'),result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
