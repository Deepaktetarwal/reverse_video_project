import csv
from pathlib import Path
from .common import *

def main():
    p=arguments('Summarize manually rated inverse-action success and visual quality.')
    p.add_argument('--split',choices=['val','test'],default='test')
    a=p.parse_args(); c=config(a.config); out=Path(c['run_dir'])/'evaluation'/a.split
    with (out/'human_review.csv').open(newline='',encoding='utf-8') as f: records=list(csv.DictReader(f))
    result={}
    for mode in ('teacher_reversed_reference','unadapted','student'):
        selected=[r for r in records if r['mode']==mode]
        rated=[]; visual=[]; temporal=[]
        for r in selected:
            s=r['reverse_action_success'].strip()
            if s:
                if s not in ('0','1'): raise ValueError('Action success must be 0 or 1')
                rated.append(int(s))
            for key,target in [('visual_quality_1to5',visual),('temporal_quality_1to5',temporal)]:
                if r[key].strip():
                    value=float(r[key])
                    if not 1<=value<=5: raise ValueError('Quality ratings must be 1–5')
                    target.append(value)
        result[mode]={'rated_action_clips':len(rated),'expected_clips':len(selected),'reverse_action_success_rate':sum(rated)/len(rated) if rated else None,'visual_quality_mean':sum(visual)/len(visual) if visual else None,'temporal_quality_mean':sum(temporal)/len(temporal) if temporal else None,'review_complete':len(rated)==len(selected) and len(visual)==len(selected) and len(temporal)==len(selected)}
    write_json(out/'human_summary.json',result)
    print(__import__('json').dumps(result,indent=2))
if __name__=='__main__': main()
