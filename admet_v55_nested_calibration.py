"""Nested scaffold evaluation: calibration estimates never see outer-query labels."""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(_k,'2')
import json
from pathlib import Path
import numpy as np
import admet_v24_models as b
import admet_v44_run as v44
from admet_v52_calibration import fit_lad

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v55_nested_calibration'


def main():
    b.remote_only();OUT.mkdir(exist_ok=True);d=b.build_data();records=[]
    for task,mode,shrink,variant in [('ppbr_az','median_bias',1.,0),('lipophilicity_astrazeneca','lad_affine',.7,1)]:
        k=d['tasks'].index(task);allrows=np.flatnonzero((d['tid']==k)&(d['split']<2));groups=np.unique(d['rg'][allrows]);parts=[]
        for seed in [531016,531017]:
            rng=np.random.default_rng(seed+k);hold=rng.permutation(groups)[:max(1,int(.25*len(groups)))]
            active=np.isin(d['rg'][allrows],hold);rr,q=allrows[~active],allrows[active]
            outer=v44.variants(v44.fit(d,k,rr,q,v44.OUT/task/f'v53_calibration_{seed}'),'MAE')[variant]
            # Estimate bias/slope from OOF predictions wholly within the outer fit rows.
            innergroups=np.unique(d['rg'][rr]);order=np.random.default_rng(551016+seed+k).permutation(innergroups)
            assigned={g:j%3 for j,g in enumerate(order)};folds=np.array([assigned[g] for g in d['rg'][rr]])
            inner=np.empty(len(rr))
            for f in range(3):
                ix=folds==f;train,query=rr[~ix],rr[ix]
                pp=v44.variants(v44.fit(d,k,train,query,v44.OUT/task/f'v55_inner_{seed}_{f}'),'MAE')[variant]
                inner[ix]=pp
            beta=np.array([1.,np.median(d['y'][rr]-inner)]) if mode=='median_bias' else fit_lad(inner,d['y'][rr])
            beta=np.array([1.,0.])+shrink*(beta-np.array([1.,0.]))
            new=beta[0]*outer+beta[1]
            if task=='ppbr_az':new=new.clip(0,100)
            gain=b.delta('MAE',d['y'][q],outer,new)
            parts.append(dict(seed=seed,fit_rows=len(rr),query_rows=len(q),coefficients=beta.tolist(),
                control_mae=b.metric('MAE',d['y'][q],outer),calibrated_mae=b.metric('MAE',d['y'][q],new),
                gain=gain,outer_query_labels_used_for_calibration=False,global_scaffold_overlap=0))
            b.emit('v55_outer',task=task,**parts[-1])
        records.append(dict(task=task,mode=mode,shrink=shrink,parts=parts,mean_gain=float(np.mean([x['gain'] for x in parts])),
            pass_gate=bool(np.mean([x['gain'] for x in parts])>0 and all(x['gain']>-.005 for x in parts))))
        b.save(OUT/'analysis.json',dict(status='IN_PROGRESS',tasks=records))
    b.save(OUT/'analysis.json',dict(status='COMPLETE',tasks=records,current_official_score=None,
        limitations='Nested coefficient estimation excludes all outer query labels. Calibration form was frozen from historical known data; source model is v44 rather than full ensemble. Repartitioned development data, not external data.'))


if __name__=='__main__':main()
