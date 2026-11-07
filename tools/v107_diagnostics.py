"""Report-only fixed-candidate uncertainty and endpoint prediction shifts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v107_delivery as delivery
def main():
    report=delivery.read(delivery.OUT/'release.json');records=[];d=delivery.count.load()
    for task,ch in report['selected'].items():
        v=delivery.BRANCHES[ch['version']];dd=v.load() if ch['version'] in (104,106) else d;r=ch['record'];k=d['tasks'].index(task)
        for part in ('fold2','valid'):
            q=v.r83.split(dd,k,part)[1];y=dd['y'][q];old=v.base(dd,k,part);new=v.mix(r['metric'],old,v.expert(dd,k,part,r['config']),r['weight'])
            groups,inv=np.unique(dd['groups'][q],return_inverse=True);rec=dict(task=task,branch=ch['version'],part=part,metric=r['metric'],rows=len(q),groups=len(groups),gain=v.r83.gain(r['metric'],y,old,new))
            if len(groups)<20:rec.update(status='NOT_ESTIMABLE_TOO_FEW_GROUPS',percentile_95=None)
            else:
                blocks=[np.flatnonzero(inv==j) for j in range(len(groups))];rng=np.random.default_rng(1074026+k+(100 if part=='valid' else 0));gains=[]
                for _ in range(1000):
                    ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
                    if r['metric'] in ('AUROC','AUPRC') and len(np.unique(y[ix]))<2:continue
                    z=v.r83.gain(r['metric'],y[ix],old[ix],new[ix])
                    if np.isfinite(z):gains.append(z)
                rec.update(status='COMPLETE',percentile_95=np.quantile(gains,[.025,.975]).tolist(),positive_fraction=float(np.mean(np.array(gains)>0)))
            records.append(rec)
    out=dict(status='COMPLETE',selection_effect='NONE_REPORT_ONLY',records=records,official_score=None,
        limitations='Historical groups and approximate baseline; labels have been repeatedly reused. Intervals do not incorporate model selection or estimate platform points.')
    delivery.count.b.save(delivery.OUT/'diagnostics.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
