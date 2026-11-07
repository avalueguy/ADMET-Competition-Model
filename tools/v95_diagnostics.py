"""Report-only scaffold bootstrap for frozen normalized-assay candidates."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v95_graph_specialists as v

def main():
    d=v.load();conf=json.loads((v.OUT/'confirmation.json').read_text());records=[]
    assert conf['status']=='COMPLETE'
    for r in conf['records']:
        if not r['accepted']:continue
        k=d['tasks'].index(r['task']);metric=r['metric']
        for part in ('fold2','valid'):
            q=v.r83.split(d,k,part)[1];old=v.base(d,k,part);p=v.mix(metric,old,v.expert(d,k,part,r['config']),r['weight']);y=d['y'][q]
            groups,inverse=np.unique(d['groups'][q],return_inverse=True)
            if len(groups)<20:
                records.append(dict(task=r['task'],part=part,metric=metric,rows=len(q),scaffolds=len(groups),
                    gain=v.r83.gain(metric,y,old,p),percentile_95=None,positive_fraction=None,
                    status='NOT_ESTIMABLE_TOO_FEW_GROUPS',reason='Historical exclusion grouping is degenerate; no confidence interval is reported.'))
                continue
            blocks=[np.flatnonzero(inverse==j) for j in range(len(groups))];rng=np.random.default_rng(950923+k+(100 if part=='valid' else 0));gains=[]
            for _ in range(1000):
                ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
                if metric in ('AUROC','AUPRC') and len(np.unique(y[ix]))<2:continue
                gain=v.r83.gain(metric,y[ix],old[ix],p[ix])
                if np.isfinite(gain):gains.append(gain)
            records.append(dict(task=r['task'],part=part,metric=metric,rows=len(q),scaffolds=len(groups),
                gain=v.r83.gain(metric,y,old,p),percentile_95=np.quantile(gains,[.025,.975]).tolist(),positive_fraction=float(np.mean(np.array(gains)>0))))
    report=dict(status='COMPLETE',selection_effect='NONE_REPORT_ONLY',records=records,
        limitations='Historically reused labels, proxy baseline, conditional official measurements; does not estimate official score.')
    v.b.save(v.OUT/'diagnostics.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
