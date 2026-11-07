"""Report-only paired scaffold bootstrap, after frozen confirmation.

Bootstrap intervals do not govern selection and do not estimate official
score changes. Historical proxies and reused labels remain limitations.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import v84_neighbor_transfer as v


def main():
    d=v.load();conf=json.loads((v.OUT/'confirmation.json').read_text())
    assert conf['status']=='COMPLETE' and conf['signature']==v.signature()
    records=[]
    for r in conf['records']:
        k=d['tasks'].index(r['task']);metric=r['metric']
        for part in ('fold2','valid'):
            q=v.old.split(d,k,part)[1]
            p=np.load(v.OUT/'models'/r['task']/part/r['config']['name']/'predictions.npz')['p']
            old=v.baseline(d,k,part);new=v.mix(metric,old,p,r['weight']);y=d['y'][q]
            groups,inverse=np.unique(d['groups'][q],return_inverse=True)
            blocks=[np.flatnonzero(inverse==g) for g in range(len(groups))]
            rng=np.random.default_rng(840912+k+(100 if part=='valid' else 0))
            gains=[]
            for _ in range(1500):
                ix=np.concatenate([blocks[i] for i in rng.integers(len(groups),size=len(groups))])
                if metric in ('AUROC','AUPRC') and len(np.unique(y[ix]))<2:continue
                g=v.old.gain(metric,y[ix],old[ix],new[ix])
                if np.isfinite(g):gains.append(g)
            record=dict(task=r['task'],part=part,metric=metric,accepted=r['accepted'],
                rows=len(q),scaffolds=len(groups),point_gain=v.old.gain(metric,y,old,new),
                replicates=len(gains),percentile_95=np.quantile(gains,[.025,.975]).tolist(),
                positive_fraction=float(np.mean(np.array(gains)>0)))
            records.append(record);v.emit(stage='bootstrap_report_only',**record)
    v.b.save(v.OUT/'uncertainty.json',dict(status='COMPLETE',selection_effect='NONE_REPORT_ONLY',
        records=records,limitations='Reused historical labels, historical proxy baseline and conditional assays; not an official score estimate.'))


if __name__=='__main__':main()
