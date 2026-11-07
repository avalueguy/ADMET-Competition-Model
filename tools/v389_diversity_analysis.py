"""Post-run development-only complementarity description, no new selection."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/regression9_v389'
def main():
    dev=json.loads((EXP/'development.json').read_text());data=np.load(ROOT/'experiments/regression9_v387/official_rows.npz');out=[]
    for r in dev['records']:
        single=max((c for c in r['choices']if c['a']==0 or c['b']==0),key=lambda x:x['objective'])
        paired=max((c for c in r['choices']if c['a']>0 and c['b']>0),key=lambda x:x['objective']);parts=[]
        for p in('fold0','fold1'):
            z=[np.load(EXP/'components'/v/r['task']/(p+'.npz'))for v in('v386','v387')];y=data['y'][z[0]['query']]
            e=[x['p']-y for x in z]
            parts.append(dict(part=p,signed_error_pearson=float(np.corrcoef(e)[0,1]),
                opposite_error_sign_fraction=float(np.mean(e[0]*e[1]<0)),prediction_disagreement_MAE=float(abs(z[0]['p']-z[1]['p']).mean())))
        out.append(dict(task=r['task'],frozen_choice={k:r['best'][k]for k in('a','b','objective')},
            best_single_development_objective=single['objective'],best_two_positive_development_objective=paired['objective'],
            two_positive_over_single_objective_gain=paired['objective']-single['objective'],development_error_diversity=parts))
    result=dict(records=out,no_fitting_performed=True,no_confirmation_reselection=True,descriptive_only=True)
    (ROOT/'docs/v389_diversity_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(two_positive_best=sum(r['frozen_choice']['a']>0 and r['frozen_choice']['b']>0 for r in out),records=[dict(task=r['task'],**r['frozen_choice'])for r in out])))
if __name__=='__main__':main()
