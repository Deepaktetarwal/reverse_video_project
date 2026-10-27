"""Build a V3 report from real saved artifacts, leaving missing results explicit."""
import argparse,json
from pathlib import Path
from .core import load_config

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True)
    a=p.parse_args();c=load_config(a.config);root=Path(c['run_dir'])
    text='# Same-prompt reverse-video refinement: V3 report\n\n'
    text+='Both teacher and student receive the original forward-action prompt. The target is the separately VAE-encoded pixel reversal of the frozen teacher sample. Student sampling and decoding preserve generated frame order.\n\n'
    text+='## Motivation and method\n\nThe preceding V2 run produced forward-action outputs in the reviewed tests. V3 uses fresh-noise conditional flow matching, normalized latent-motion reconstruction, and a proposed reverse-versus-forward endpoint hinge. It tests one training clip before a full run. No reverse-caption rewrite is used. See docs/REFINEMENT_MATH_V3.md for equations, assumptions and limitations.\n\n'
    text+='Default kd_weight=0 is teacher-sample distillation. The optional separate transported-velocity-KD ablation must be reported separately. Changes in learning rate, rank, noise sampling and loss mean that this experiment alone does not isolate the cause of the earlier failure.\n\n'
    text+='## Exact configuration\n\n```json\n'+json.dumps(c,indent=2)+'\n```\n\n'
    artifacts=['gpu_smoke_v3.json','baseline_validation.json','training_complete.json','best.json','latest.json','diagnostics/train_student.json','diagnostics/val_student.json','diagnostics/test_student.json','diagnostics/test_unadapted.json','evaluation/test/metrics_summary.json','evaluation/test/human_summary.json']
    for name in artifacts:
        text+='## '+name+'\n\n'
        path=root/name
        text+=('```json\n'+path.read_text()+'\n```\n\n') if path.exists() else 'NOT RUN / NOT AVAILABLE\n\n'
    text+='## Training log\n\n'
    log=root/'training_log_v3.jsonl'
    if log.exists():
        records=[json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        bystep={r['step']:r for r in records}
        text+=f'{len(bystep)} distinct logged updates; highest logged step {max(bystep) if bystep else 0}. Logs may include repeated uncommitted steps after a disconnect. Latest committed checkpoint is identified above.\n\n'
    else:text+='NOT RUN / NOT AVAILABLE\n\n'
    text+='## Qualitative assessment and discussion — COMPLETE FROM OBSERVATIONS\n\nRecord whether forward/reverse VAE reconstructions retain the intended action. Show both fixed diagnostic seeds, validation comparisons and every final test clip, including failures. Record visible inverse-action completion rather than relying on reconstruction proxies. Explain checkpoint selection using validation, settings/compute, experiments, and failure cases.\n\n'
    text+='## Interface and submission\n\nnotebooks/Wan_Reverse_Refine_V3_Colab.ipynb section 15 takes an original prompt and seed, uses saved original embeddings or a frozen text encoder, runs the trained LoRA transformer and decodes frames without reversing them. It displays video and saved human metrics if available. Submit source, dependency pins, adapter/base revision, exact configuration, training log, diagnostics, videos, completed ratings and this completed report.\n\n'
    text+='## Limits\n\nOnly 32 training clips and four test contexts; test clips were already inspected in V2. The endpoint hinge can exploit non-motion appearance or causal-boundary differences. One-clip success can be memorization. LoRA keeps the full base architecture and does not imply inference acceleration. Full-model GPU execution and reverse-motion quality must be established by the saved experiment, not by the CPU component tests.\n'
    out=root/'REPORT_V3.md';out.write_text(text);print(out)
if __name__=='__main__':main()
