"""Reuse unchanged VAE/KD caches; never copy v1 learned adapters or optimizer state."""
from pathlib import Path
import shutil,os
from .common import *
from .conditioning import MODE

def migrate(c,source):
    source=Path(source); destination=Path(c['run_dir'])
    if source.resolve()==destination.resolve(): raise ValueError('Use a NEW run_dir for original-prompt training.')
    receipt=destination/'migration_v2.json'
    if receipt.exists():
        print('Cache migration already completed. Existing new-run progress is preserved.',flush=True)
        return read_json(receipt)
    signature=prep_signature(c,manifest(c))
    expected={'prepared':signature,'kd_cache':cache_signature(c,signature)}
    copied={}
    for name in ('prepared','kd_cache'):
        src=source/name; dst=destination/name
        if not (src/'metadata.json').exists():
            copied[name]=0
            print('No saved',name,'found; the normal stage will create it.',flush=True)
            continue
        meta=read_json(src/'metadata.json')
        if meta['signature']!=expected[name]: raise ValueError('Source '+name+' differs from this config/data. Copy the original config settings or run a fresh preparation.')
        dst.mkdir(parents=True,exist_ok=True)
        committed=[]
        for tensor in sorted(src.glob('*.safetensors')):
            marker=tensor.with_suffix('.json')
            if not marker.exists(): continue
            if sha256(tensor)!=read_json(marker)['sha256']: raise ValueError('Damaged old cache: '+str(tensor))
            committed.append((tensor,marker))
        files=[tensor for tensor,marker in committed]+[marker for tensor,marker in committed]
        if (src/'index.jsonl').exists(): files.append(src/'index.jsonl')
        files.append(src/'metadata.json')
        for file in files:
            target=dst/file.name
            if target.exists():
                if sha256(target)!=sha256(file): raise ValueError('Migration conflict: '+str(target)+'. Do not overwrite ongoing new-run work.')
                continue
            tmp=target.with_name(target.name+'.copying')
            shutil.copy2(file,tmp)
            if sha256(tmp)!=sha256(file): raise RuntimeError('Cache copy verification failed')
            os.replace(tmp,target)
        copied[name]=len(committed)
        print('Reused',name,len(committed),'committed tensors.',flush=True)
    result={'conditioning':MODE,'source_run':str(source),'destination_run':str(destination),'copied_tensors':copied,'note':'Only prepared and frozen-teacher KD files copied. New smoke and fresh adapter training required.'}
    write_json(receipt,result)
    return result

def main():
    p=arguments(__doc__);p.add_argument('--from-run',required=True)
    a=p.parse_args();c=config(a.config)
    print(__import__('json').dumps(migrate(c,a.from_run),indent=2))
if __name__=='__main__': main()
