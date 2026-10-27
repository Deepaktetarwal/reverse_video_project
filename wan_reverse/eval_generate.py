import os,subprocess,sys
from pathlib import Path
import numpy as np
from .common import *

def main():
    p=arguments('Generate fixed held-out teacher-reference, unadapted and trained-student comparisons.')
    p.add_argument('--checkpoint',default='best'); p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--seed',type=int,default=2026); p.add_argument('--limit',type=int,default=0)
    a=p.parse_args(); c=config(a.config)
    source=[r for r in manifest(c) if r['split']==a.split]
    if a.limit>0: source=source[:a.limit]
    out=Path(c['run_dir'])/'evaluation'/a.split; out.mkdir(parents=True,exist_ok=True)
    generated=[]
    from .prepare import preprocess
    from .decode import write_video
    for row in source:
        root=out/row['id']; root.mkdir(exist_ok=True)
        path=safe_asset(c['dataset_dir'],row['assets']['frames_uint8']['path'])
        if sha256(path)!=row['assets']['frames_uint8']['sha256']: raise ValueError('Reference checksum mismatch')
        with np.load(path,allow_pickle=False) as f: x,indices=preprocess(f['frames'],c)
        # Frame reversal is restricted to constructing the TEACHER reference.
        pixels=np.rint(x[0].permute(1,2,3,0).numpy()[::-1].copy()*255).astype(np.uint8)
        reference=root/'teacher_reversed_reference.mp4'; write_video(reference,pixels,c['fps'])
        modes={}
        for mode in ('unadapted','student'):
            mode_root=root/mode
            command=[sys.executable,'-m','wan_reverse.sample','--config',a.config,'--mode',mode,'--id',row['id'],'--seed',str(a.seed),'--checkpoint',a.checkpoint,'--out-dir',str(mode_root)]
            subprocess.run(command,check=True)
            marker=read_json(mode_root/'last_sample.json')['metadata']
            subprocess.run([sys.executable,'-m','wan_reverse.decode','--config',a.config,'--sample',marker],check=True)
            modes[mode]=str(Path(marker).with_suffix('.mp4'))
        generated.append({'id':row['id'],'event':row['event'],'split':row['split'],'seed':a.seed,'prompt':row['prompt'],'reverse_caption':row['reverse_caption'],'teacher_reversed_reference':str(reference),**modes})
        write_rows(out/'comparison_manifest.jsonl',generated)
    print('Comparison manifest:',out/'comparison_manifest.jsonl',flush=True)
if __name__=='__main__': main()
