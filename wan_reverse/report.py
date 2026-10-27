from pathlib import Path
import json
from .common import *

def optional(path):
    return read_json(path) if Path(path).exists() else {'status':'NOT RUN / NOT AVAILABLE'}

def main():
    a=arguments('Build a stage-wise report populated only with existing experiment artifacts.').parse_args()
    c=config(a.config); root=Path(c['run_dir']); out=root/'REPORT.md'
    sections=[('# Reverse-time Wan distillation: experiment report','This report is generated from saved artifacts. Missing stages remain marked as not run. Human conclusions must be completed after watching the held-out results.'),
      ('1. Objective','Adapt a frozen pretrained Wan2.1 text-to-video model into a LoRA student that directly denoises reverse-action video latents. Student inference does not generate a forward video and flip frames.'),
      ('2. Teacher and dataset','Teacher: '+c['model_id']+' at '+c['revision']+'. The existing teacher candidates were generated at 480×832, 81 frames and 50 steps. Four event families are falling, pouring, ink dispersion and balloon inflation. The fixed split is 32 train / 4 validation / 4 test contexts. Reverse pixel sequences provide targets and the original forward-action captions provide conditioning for both models; reverse captions are evaluation labels only. Acceptance in this run is user-declared if the corresponding config flag is true; it is not an independent quality judgment.'),
      ('3. Data audit',json.dumps(optional(root/'audit.json'),indent=2)),
      ('4. Preprocessing','Uniformly select '+str(c['frames'])+' frames covering both endpoints of the full stored action and resize to '+str(c['height'])+'×'+str(c['width'])+'. Encode the forward pixel tensor and its frame reversal in separate causal Wan VAE calls, using posterior modes and the pretrained per-channel mean/std normalization. This changes training resolution and temporal density relative to the original generation. Display at '+str(c['fps'])+' fps; it does not preserve native playback speed.'),
      ('5. Distillation method','Use shared Gaussian noise and sigma for paired forward/reverse latents. xf=(1-s)zf+sε and xr=(1-s)zr+sε. Teacher target is vT(xf,s,c)+zf-zr, derived from the pair-dependent translation xr=xf+(1-s)(zr-zf). Student conditions on the same original forward-action caption as the teacher. Also minimize exact conditional flow target ε-zr and a latent temporal reconstruction term. This is a proposed paired transport objective, not an established guarantee of semantic reverse-time learning. See docs/MATHEMATICS_WAN.md.'),
      ('6. Hardware execution test',json.dumps(optional(root/'gpu_smoke.json'),indent=2)),
      ('7. Training configuration',json.dumps(c,indent=2)),
      ('8. Training result',json.dumps(optional(root/'training_complete.json'),indent=2)),
      ('9. Selected checkpoint',json.dumps(optional(root/'best.json'),indent=2)),
      ('10. Held-out losses',json.dumps({m:optional(root/'evaluation/test'/f'{m}_heldout_loss.json') for m in ('unadapted','student')},indent=2)),
      ('11. Video proxy metrics',json.dumps(optional(root/'evaluation/test/metrics_summary.json'),indent=2)),
      ('12. Human visual and inverse-action evaluation',json.dumps(optional(root/'evaluation/test/human_summary.json'),indent=2)),
      ('13. Interface','The final cells of notebooks/Wan_Reverse_SamePrompt_Colab.ipynb provide a text prompt, a cached example selector, random seed and a Generate button. A cached original prompt reuses its frozen forward-caption embedding; new text is encoded by frozen quantized UMT5 in a separate process. The trained adapter and base transformer directly generate latent frames; a separate VAE process decodes them in model order. The interface displays saved video and measured time. It requires a completed trained checkpoint.'),
      ('14. Discussion to complete','Describe which inverse events actually complete, compare against the unadapted generator, and record failure cases without changing test prompts or selecting only favorable test seeds. Do not infer successful reverse learning from decreasing train loss alone. Insert representative qualitative stills/videos and explain each major experiment.'),
      ('15. Limitations','Only 32 training clips and four test contexts; generated targets inherit teacher errors; LoRA retains the full base architecture and does not reduce its inference cost; the pair-dependent translation is a modeling choice; cached noise draws have limited diversity; quantized text encoding can differ from full precision; latent temporal loss is not a pixel-space physics constraint; CLIP is largely insensitive to frame order; low warp error can reward static output; train resolution/frame count differs from original 480p/81-frame teacher generations; full T4 training memory and runtime must be established by the saved smoke test.'),
      ('16. Reproducibility and submission','Submit this source package, pinned dependency list, exported adapter with base revision and rank/alpha, checkpoints/training logs, exact config, dataset manifests/review declaration, held-out videos, quantitative reports and completed human ratings. No empirical result should be copied into the report before the corresponding artifact exists.')]
    text=''
    for title,body in sections:
        text+=(title if title.startswith('#') else '## '+title)+'\n\n'
        if not title.startswith('#'): pass
        if body.startswith('{'): text+='```json\n'+body+'\n```\n\n'
        else: text+=body+'\n\n'
    out.write_text(text,encoding='utf-8'); print(out)
if __name__=='__main__': main()
