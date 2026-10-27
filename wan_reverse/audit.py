from collections import Counter
from pathlib import Path
import numpy as np
from .common import *

def main():
    p=arguments('Validate imported Wan dataset; no GPU needed.')
    p.add_argument('--verify-all',action='store_true',help='Hash all video/latent assets as well as training inputs')
    a=p.parse_args(); c=config(a.config); data=manifest(c); root=Path(c['dataset_dir'])
    selected=['frames_uint8','embedding','negative_embedding']
    seen={}
    for i,r in enumerate(data):
        keys=r['assets'] if a.verify_all else selected
        for key in keys:
            asset=r['assets'][key]; path=safe_asset(root,asset['path'])
            if str(path) in seen:
                if seen[str(path)]!=asset['sha256']: raise ValueError('Conflicting shared asset hash')
                continue
            if not path.is_file() or path.stat().st_size!=asset['bytes'] or sha256(path)!=asset['sha256']:
                raise ValueError(f'Missing/damaged asset: {path}')
            seen[str(path)]=asset['sha256']
        with np.load(safe_asset(root,r['assets']['frames_uint8']['path']),allow_pickle=False) as f:
            x=f['frames']
            if x.dtype!=np.uint8 or x.ndim!=4 or x.shape[-1]!=3: raise ValueError('Expected THWC uint8 frames')
            if x.shape[0]<c['frames']: raise ValueError('Not enough frames for requested preprocessing')
            if c.get('require_all_81_frames') and x.shape[0]!=81:raise ValueError('Expected exactly 81 stored source frames')
        print(f"[{i+1}/{len(data)}] {r['id']} OK",flush=True)
    counts=dict(Counter(r['split'] for r in data))
    if not all(counts.get(s,0)>0 for s in ('train','val','test')): raise ValueError('Every split must contain data')
    if c.get('expected_candidates')==40 and counts!={'train':32,'val':4,'test':4}: raise ValueError('Expected fixed 32/4/4 split')
    result={'clips':len(data),'splits':counts,'events':dict(Counter(r['event'] for r in data)),'user_declared_all_accepted':c['accept_all_user_declared'],'training_resolution':[c['height'],c['width']],'training_frames':c['frames'],'signature':prep_signature(c,data),'all_assets_hashed':a.verify_all}
    write_json(Path(c['run_dir'])/'audit.json',result)
    print(__import__('json').dumps(result,indent=2))
if __name__=='__main__': main()
