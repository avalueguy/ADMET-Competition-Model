"""Selected final ensemble uncertainty, retaining search-bias limitations."""
import json
import numpy as np
import admet_v24_models as b
import admet_v33_run as n
from admet_v36_run import ROOT,OUT,signature,source_predictions
def main():
    b.remote_only();d=b.build_data();report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    base=np.load(n.OUT/'baseline_valid.npy');folds=np.load(b.OUT/'folds.npy');rng=np.random.default_rng(360699);records=[]
    for r in report['tasks']:
        s=r['selection']
        if s is None:continue
        t=r['task'];k=d['tasks'].index(t);m=r['metric'];vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        p=np.zeros(len(vi))
        for source in s['sources']:
            a=source_predictions(d,t,source['version'],source['variant'],'valid')
            if s['name']=='complement_equal' and m!='MAE':a=b.rank(a)
            p+=source['weight']*a
        merged=b.blend(m,base[vi],p,s['weight']);gain=b.delta(m,d['y'][vi],base[vi],merged)
        np.testing.assert_allclose(gain,s['valid_gain'],rtol=1e-9,atol=1e-10)
        groups,iv=np.unique(d['rg'][vi],return_inverse=True);blocks=[np.flatnonzero(iv==j) for j in range(len(groups))];draws=[]
        for _ in range(400):
            ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
            if m in ['AUROC','AUPRC'] and len(np.unique(d['y'][vi][ix]))<2:continue
            draws.append(b.delta(m,d['y'][vi][ix],base[vi][ix],merged[ix]))
        held=[]
        for fold in range(3):
            choices=[g for g in r['grid'] if np.mean([g['cv_gains'][j] for j in range(3) if j!=fold])>5e-4]
            best=max(choices,key=lambda g:np.mean([g['cv_gains'][j] for j in range(3) if j!=fold])) if choices else None
            held.append(best['cv_gains'][fold] if best else 0.)
        records.append(dict(task=t,metric=m,valid_gain=gain,valid_rows=len(vi),bootstrap_scaffolds=len(groups),
            post_selection_scaffold_bootstrap_95=np.quantile(draws,[.025,.975]).tolist(),
            bootstrap_positive_fraction=float(np.mean(np.array(draws)>0)),
            held_fold_source_selection_gains=held,held_fold_mean=float(np.mean(held))))
    b.save(OUT/'uncertainty.json',dict(signature=signature(),tasks=records,
        limitation='Bootstrap is post-selection over reused valid and does not account for model search. Held-fold selection chooses among already searched source winners; not fully nested model-training validation.'))
if __name__=='__main__':main()
