"""Export trained LoRA plus reconstruction metadata; base model downloaded separately."""
import shutil,zipfile
from pathlib import Path
from .common import *
from .training import resolve_checkpoint,verify_checkpoint

def main():
    p=arguments('Package trained adapter, configuration, evidence and source provenance.')
    p.add_argument('--checkpoint',default='best')
    a=p.parse_args(); c=config(a.config); path=resolve_checkpoint(c,a.checkpoint); m=verify_checkpoint(c,path)
    if m['step']<1: raise ValueError('Train first')
    out=Path(c['run_dir'])/'export'; out.mkdir(exist_ok=True)
    shutil.copy2(path/'adapter.safetensors',out/'adapter.safetensors')
    shutil.copy2(path/'metadata.json',out/'checkpoint_metadata.json')
    write_json(out/'config.json',c)
    (out/'USAGE.txt').write_text('Use the supplied wan_reverse code and matching config. Base: '+c['model_id']+' at '+c['revision']+'. Load the base transformer, inject custom LoRA with the saved rank/alpha, then load adapter.safetensors. This is a LoRA checkpoint, not a full/compressed base model. Run wan_reverse.sample followed by wan_reverse.decode. Supply the ORIGINAL forward-action prompt unchanged; the student is trained to produce its inverse action. The decoder preserves model frame order.\n',encoding='utf-8')
    files=[p for p in out.iterdir() if p.is_file() and p.suffix!='.zip']
    for name in ('audit.json','gpu_smoke.json','training_complete.json','training_log.jsonl','gpu_smoke_v3.json','training_log_v3.jsonl','baseline_validation.json','REPORT_V3.md'):
        p=Path(c['run_dir'])/name
        if p.exists(): files.append(p)
    zip_path=out/'wan_reverse_trained_adapter.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files: z.write(p,p.name)
        diagnostics=Path(c['run_dir'])/'diagnostics'
        if diagnostics.exists():
            for p in diagnostics.glob('*.json'): z.write(p,'diagnostics/'+p.name)
        evaluation=Path(c['run_dir'])/'evaluation'
        if evaluation.exists():
            for p in evaluation.rglob('*'):
                if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.html'):
                    z.write(p,'evaluation/'+p.relative_to(evaluation).as_posix())
    print('Export:',zip_path,flush=True)
if __name__=='__main__': main()
