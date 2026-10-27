from pathlib import Path
from safetensors.torch import load_file
from .common import *
from .models import transformer
from .adapters import load_adapter
from .training import resolve_checkpoint,verify_checkpoint,validation

def main():
    p=arguments('Fixed held-out flow losses: compare unadapted and trained Wan.')
    p.add_argument('--mode',choices=['student','unadapted'],default='student'); p.add_argument('--checkpoint',default='best')
    p.add_argument('--split',choices=['val','test'],default='test')
    a=p.parse_args(); c=config(a.config); meta,entries=cache_index(c)
    if not meta['complete']: raise ValueError('Finish cache first')
    stage_training_files(c)
    model,_=transformer(c,student=a.mode=='student')
    if a.mode=='student':
        path=resolve_checkpoint(c,a.checkpoint); verify_checkpoint(c,path); load_adapter(model,load_file(str(path/'adapter.safetensors')))
    result=validation(model,c,[r for r in entries if r['split']==a.split])
    out=Path(c['run_dir'])/'evaluation'/a.split; out.mkdir(parents=True,exist_ok=True)
    write_json(out/(a.mode+'_heldout_loss.json'),{'mode':a.mode,'split':a.split,'entries':sum(r['split']==a.split for r in entries),'losses':result})
    print(result)
if __name__=='__main__': main()
