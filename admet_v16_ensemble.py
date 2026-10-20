"""Primary-metric convex stacking; no per-metric non-regression vetoes."""
import numpy as np
from scipy.stats import rankdata
from itertools import product
from admet_v6_runner import metrics,kind,deltas


def combine(base,candidates,weights,mode,metric):
    base=np.asarray(base,float);x=np.column_stack([base,*candidates]);w=np.asarray(weights,float)
    assert len(w)==x.shape[1] and np.min(w)>=-1e-10 and abs(w.sum()-1)<1e-9
    if mode=='raw':return x@w
    assert mode=='rank' and metric!='MAE'
    n=len(base);r=np.column_stack([(rankdata(x[:,j],method='average')-.5)/n for j in range(x.shape[1])]);q=r@w
    return q if metric in ['AUROC','AUPRC'] else np.interp(q,(np.arange(n)+.5)/n,np.sort(base))


def options(count,grid):
    assert 1<=count<=3
    return [[round(1-sum(x),10),*x] for x in product(grid,repeat=count) if sum(x)<=1+1e-9]



def score(y,base,new,metric):return deltas(metric,metrics(kind(metric),y,base),metrics(kind(metric),y,new))


def select(y,fold,base,candidates,yv,bv,cv,metric,p,fold_subset=None):
    folds=sorted(np.unique(fold)) if fold_subset is None else fold_subset
    results=[]
    modes=['raw'] if metric=='MAE' else p['modes']
    for weights in options(len(candidates),p['grid']):
        for mode in modes:
            # Keep the exact anchor in the zero-candidate option.
            if weights[0]==1 and mode=='rank':continue
            gains=[]
            for f in folds:
                mask=fold==f
                q=combine(base[mask],[c[mask] for c in candidates],weights,mode,metric)
                gains.append(score(y[mask],base[mask],q,metric))
            qv=combine(bv,cv,weights,mode,metric);vg=score(yv,bv,qv,metric)
            z=p['objective'];objective=z['oof_weight']*np.mean(gains)+z['official_valid_weight']*vg-z['fold_standard_deviation_penalty']*np.std(gains)-z['weight_penalty']*(1-weights[0])
            results.append(dict(weights=weights,mode=mode,objective=float(objective),mean_oof_gain=float(np.mean(gains)),
                valid_gain=float(vg),fold_gains=list(map(float,gains)),candidate_mass=float(1-weights[0])))
    return max(results,key=lambda r:r['objective']),results
