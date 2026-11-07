"""Describe fixed development learning curves; no additional selection or fits."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def main():
    exp=ROOT/'experiments/regression9_v387'
    dev=json.loads((exp/'outputs/development.json').read_text(encoding='utf-8'))
    out=[]
    for r in dev['records']:
        best=r['best'];same=[]
        for horizon in(0,2,5,8):
            c=next(x for x in r['choices']if x['config']['base']==best['config']['base']and x['config']['round']==horizon and x['weight']==best['weight'])
            same.append(dict(round=horizon,config=c['config'],weight=c['weight'],objective=c['objective'],parts=c['parts']))
        late=all(same[1]['parts'][i]['gain']>same[0]['parts'][i]['gain']for i in(0,1))
        out.append(dict(task=r['task'],frozen_choice=best['config'],weight=best['weight'],fixed_base_weight_by_round=same,round2_better_on_both_development_parts=late))
    result=dict(signature=dev['signature'],descriptive_only=True,no_confirmation_reselection=True,no_fitting=True,
        records=out,limitation='Existing public partitions reused; comparisons are descriptions of the completed budget, not independent estimates.')
    (ROOT/'docs/v387_curve_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tasks=len(out),round2_better_on_both_development_parts=sum(r['round2_better_on_both_development_parts']for r in out))))
if __name__=='__main__':main()
