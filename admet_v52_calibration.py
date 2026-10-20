"""Low-dimensional, cross-fitted MAE calibration with held-scaffold selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import admet_v24_models as b

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_model_outputs_v52_calibration'


def fit_lad(p,y,bound=None):
    # min sum absolute residuals, with a positive slope for stable extrapolation.
    n=len(y)
    from scipy.sparse import csr_matrix, eye, hstack, vstack
    x=csr_matrix(np.column_stack([p,np.ones(n)]));e=eye(n,format='csr')
    matrix=vstack([hstack([x,-e]),hstack([-x,-e])]).tocsr()
    result=linprog(np.r_[0.,0.,np.ones(n)],A_ub=matrix,b_ub=np.r_[y,-y],
        bounds=[(.7,1.3),(-10,10)]+[(0,None)]*n,method='highs',options={'threads':2})
    assert result.success,result.message
    return result.x[:2]


def main():
    assert os.name!='nt'
    OUT.mkdir(exist_ok=True);d=b.build_data();f=np.load(b.OUT/'folds.npy')
    # All five MAE task values are unchanged between v45 and v46.
    pred=np.load(ROOT/'admet_model_outputs_v46/baseline_cv_proxy.npy')
    records=[]
    for k,t in enumerate(d['tasks']):
        if d['metrics'][k]!='MAE':continue
        tr=np.flatnonzero((d['tid']==k)&(d['split']==0));p=pred[tr];y=d['y'][tr];ff=f[tr]
        bounds=(0,100) if t=='ppbr_az' else None
        grid=[];cvp={}
        for mode in ['median_bias','lad_affine']:
            for shrink in [.35,.7,1.0]:
                z=np.empty_like(p);pars=[]
                for fold in range(3):
                    a=ff!=fold;q=~a
                    beta=np.array([1.,np.median(y[a]-p[a])]) if mode=='median_bias' else fit_lad(p[a],y[a])
                    beta=np.array([1.,0.])+shrink*(beta-np.array([1.,0.]))
                    z[q]=beta[0]*p[q]+beta[1]
                    if bounds:z[q]=z[q].clip(*bounds)
                    pars.append(beta.tolist())
                gs=[b.delta('MAE',y[ff==fold],p[ff==fold],z[ff==fold]) for fold in range(3)]
                row=dict(mode=mode,shrink=shrink,fold_gains=gs,mean_gain=float(np.mean(gs)),coefficients=pars)
                grid.append(row);cvp[(mode,shrink)]=z
        # Choosing mode on two evaluation folds and testing on the omitted fold still uses
        # base OOF data; descriptive diagnostic, not full nested retraining of the ensemble.
        held=[]
        for fold in range(3):
            r=max(grid,key=lambda x:np.mean([g for j,g in enumerate(x['fold_gains']) if j!=fold]))
            held.append(r['fold_gains'][fold])
        best=max(grid,key=lambda x:x['mean_gain'])
        records.append(dict(task=t,grid=grid,best=best,held_gains=held,held_mean=float(np.mean(held))))
        print(json.dumps(records[-1]),flush=True)
    b.save(OUT/'analysis.json',dict(status='DIAGNOSTIC_ONLY',tasks=records,
        source_sha256=b.sha(ROOT/'admet_v52_calibration.py'),test_labels_used=False,current_platform_score=None,
        limitations='Calibration on repeatedly reused ensemble OOF proxies. Not a new base model. Nested mode selection does not reconstruct full ensemble OOF.'))


if __name__=='__main__':main()
