"""New scaffold partitions for calibration; refit source models before using holdout labels."""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(_k,'2')
import json
from pathlib import Path
import numpy as np
import admet_v24_models as b
import admet_v44_run as v44
from admet_v52_calibration import fit_lad

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_model_outputs_v53_calibration_verify'


def main():
    b.remote_only();OUT.mkdir(exist_ok=True);d=b.build_data()
    diag=json.loads((ROOT/'admet_model_outputs_v52_calibration/analysis.json').read_text())
    records=[]
    for task in ['ppbr_az','lipophilicity_astrazeneca']:
        k=d['tasks'].index(task);old=next(x for x in diag['tasks'] if x['task']==task);s=old['best']
        tr=np.flatnonzero((d['tid']==k)&(d['split']==0));source=np.load(ROOT/'admet_model_outputs_v46/baseline_cv_proxy.npy')[tr]
        beta=np.array([1.,np.median(d['y'][tr]-source)]) if s['mode']=='median_bias' else fit_lad(source,d['y'][tr])
        beta=np.array([1.,0.])+s['shrink']*(beta-np.array([1.,0.]))
        allrows=np.flatnonzero((d['tid']==k)&(d['split']<2));groups=np.unique(d['rg'][allrows]);gains=[];parts=[]
        for seed in [531016,531017]:
            rng=np.random.default_rng(seed+k);hold=rng.permutation(groups)[:max(1,int(.25*len(groups)))]
            active=np.isin(d['rg'][allrows],hold);rr,q=allrows[~active],allrows[active]
            preds=v44.variants(v44.fit(d,k,rr,q,v44.OUT/task/f'v53_calibration_{seed}'),'MAE')
            # Use the same source branch as the actual v44 submission where available.
            info=json.loads((v44.OUT/'analysis.json').read_text());oldsel=next(x for x in info['tasks'] if x['task']==task)['selection']
            base=preds[oldsel['variant']] if oldsel else preds[0]
            new=beta[0]*base+beta[1]
            if task=='ppbr_az':new=new.clip(0,100)
            gain=b.delta('MAE',d['y'][q],base,new);gains.append(gain)
            parts.append(dict(seed=seed,fit_rows=len(rr),query_rows=len(q),base_mae=b.metric('MAE',d['y'][q],base),
                calibrated_mae=b.metric('MAE',d['y'][q],new),gain=gain,scaffold_overlap=0))
        records.append(dict(task=task,coefficients=beta.tolist(),selection=s,parts=parts,
            mean_gain=float(np.mean(gains)),pass_gate=bool(np.mean(gains)>0 and all(g>-.005 for g in gains))))
        b.save(OUT/'analysis.json',dict(status='IN_PROGRESS',tasks=records,current_official_score=None))
        b.emit('calibration_fresh',**records[-1])
    b.save(OUT/'analysis.json',dict(status='COMPLETE',tasks=records,current_official_score=None,
        limitation='Fresh partitions use previously seen development data. Source refit is v44, not full ensemble. Calibration coefficients frozen from historical CV proxy.'))


if __name__=='__main__':main()
