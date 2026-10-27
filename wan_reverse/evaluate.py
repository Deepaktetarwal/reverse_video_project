import csv,html
from pathlib import Path
import numpy as np
from .common import *

ACTION_CAPTIONS={
 'fall':'A ball rises from the floor onto a table.',
 'pour':'Liquid flows upward from a receiving glass back into a bottle or pitcher.',
 'ink':'A dispersed ink cloud contracts into a small concentrated drop in clear water.',
 'inflate':'An inflated balloon shrinks into a small wrinkled balloon while its neck stays fixed.'}

def flow_metrics(frames):
    import cv2
    frames=np.stack([cv2.resize(f,(224,128)) for f in frames])
    errors=[]; magnitudes=[]
    yy,xx=np.mgrid[:128,:224].astype(np.float32)
    for previous,current in zip(frames[:-1],frames[1:]):
        pg=cv2.cvtColor(previous,cv2.COLOR_RGB2GRAY); cg=cv2.cvtColor(current,cv2.COLOR_RGB2GRAY)
        # Backward flow at current pixels maps into the previous frame for reconstruction.
        backward=cv2.calcOpticalFlowFarneback(cg,pg,None,0.5,3,15,3,5,1.2,0)
        mx=xx+backward[...,0]; my=yy+backward[...,1]
        warped=cv2.remap(previous,mx,my,cv2.INTER_LINEAR)
        valid=(mx>=0)&(mx<223)&(my>=0)&(my<127)
        if valid.any(): errors.append(float(np.abs(warped.astype(np.float32)-current.astype(np.float32))[valid].mean()/255))
        magnitudes.append(float(np.linalg.norm(backward,axis=-1).mean()))
    return {'warp_l1':float(np.mean(errors)) if errors else 0.,'flow_magnitude_pixels':float(np.mean(magnitudes)) if magnitudes else 0.}

def main():
    p=arguments('Video proxies and blinded human review table; no unreliable four-sample FVD.')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--skip-clip',action='store_true',help='Run flow metrics without downloading CLIP')
    a=p.parse_args(); c=config(a.config)
    out=Path(c['run_dir'])/'evaluation'/a.split
    comparisons=rows(out/'comparison_manifest.jsonl')
    if not comparisons: raise ValueError('Generate comparisons first')
    clip_model=processor=None
    if not a.skip_clip:
        import torch
        from transformers import CLIPModel,CLIPProcessor
        from huggingface_hub import model_info
        model_id='openai/clip-vit-base-patch32'
        revision_file=out/'clip_revision.json'
        if not revision_file.exists(): write_json(revision_file,{'model_id':model_id,'revision':model_info(model_id).sha})
        clip_revision=read_json(revision_file)['revision']
        processor=CLIPProcessor.from_pretrained(model_id,revision=clip_revision)
        clip_model=CLIPModel.from_pretrained(model_id,revision=clip_revision).eval().to('cpu')
    metrics=[]; cards=[]
    modes=('teacher_reversed_reference','unadapted','student')
    for row in comparisons:
        captions=ACTION_CAPTIONS[row['event']]
        for mode in modes:
            video=Path(row[mode])
            with np.load(video.with_suffix('.npz'),allow_pickle=False) as f: frames=f['frames']
            result={'id':row['id'],'event':row['event'],'mode':mode,**flow_metrics(frames),'clip_text_cosine':None,'metric_caption':captions}
            if clip_model is not None:
                import torch
                from PIL import Image
                chosen=np.linspace(0,len(frames)-1,min(8,len(frames))).round().astype(int)
                inputs=processor(text=[captions],images=[Image.fromarray(frames[i]) for i in chosen],return_tensors='pt',padding=True,truncation=True)
                with torch.inference_mode():
                    image=clip_model.get_image_features(pixel_values=inputs['pixel_values'])
                    text=clip_model.get_text_features(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
                    image=image/image.norm(dim=-1,keepdim=True); text=text/text.norm(dim=-1,keepdim=True)
                    result['clip_text_cosine']=float((image@text.T).mean())
            metrics.append(result)
        cards.append(f"<article><h2>{html.escape(row['id'])}</h2><p><b>Input to BOTH models:</b> {html.escape(row['prompt'])}</p><p><b>Expected inverse event (evaluation label only):</b> {html.escape(row['reverse_caption'])}</p>"+''.join(f"<div><b>{mode}</b><br><video controls preload='none' width='320' src='{Path(row[mode]).relative_to(out).as_posix()}'></video></div>" for mode in modes)+'</article>')
    write_json(out/'metrics_per_clip.json',metrics)
    with (out/'metrics_per_clip.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    summary={}
    for mode in modes:
        selected=[r for r in metrics if r['mode']==mode]
        summary[mode]={'n':len(selected)}
        for key in ('warp_l1','flow_magnitude_pixels','clip_text_cosine'):
            vals=[r[key] for r in selected if r[key] is not None]
            summary[mode][key]=float(np.mean(vals)) if vals else None
    write_json(out/'metrics_summary.json',{'metrics':summary,'limitations':['CLIP averaged over frames is not a temporal direction test.','Low warp error can reward a static video; inspect flow magnitude and action success together.','Optical flow includes camera motion, occlusions and lighting changes.','Only four test contexts: do not treat the averages as robust population estimates.','Teacher reference is a resized, temporally subsampled stored video; it is not a new matched-noise generation.']})
    review=out/'human_review.csv'; previous={}
    if review.exists():
        with review.open(newline='',encoding='utf-8') as f: previous={(r['id'],r['mode']):r for r in csv.DictReader(f)}
    fields=['id','mode','reverse_action_success','visual_quality_1to5','temporal_quality_1to5','notes']
    with review.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for row in comparisons:
            for mode in modes:
                w.writerow(previous.get((row['id'],mode),{'id':row['id'],'mode':mode,'reverse_action_success':'','visual_quality_1to5':'','temporal_quality_1to5':'','notes':'UNREVIEWED'}))
    (out/'comparison.html').write_text("<!doctype html><meta charset='utf-8'><title>Reverse-time comparison</title><h1>Held-out comparisons</h1><p>Complete human_review.csv. A video must visibly complete the inverse action to count as successful.</p>"+'\n'.join(cards),encoding='utf-8')
    print(__import__('json').dumps(summary,indent=2))
    print('Review sheet:',review)
if __name__=='__main__': main()
