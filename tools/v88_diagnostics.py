"""Report-only paired scaffold bootstrap for the frozen v88 release."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import v88_context_kernel as v
import v88_delivery as delivery


def main():
    d=v.load();choices=delivery.selected();records=[];movements=[]
    release=json.loads((v.OUT/'release.json').read_text())
    assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM'
    original=delivery.read(v.ANCHOR);actual=delivery.read(ROOT/release['file'])
    for r in choices:
        k=d['tasks'].index(r['task']);metric=r['metric']
        for part in ('fold2','valid'):
            q=v.r83.split(d,k,part)[1];base=v.baseline(d,k,part)
            pred=v.mix(metric,base,v.expert(d,k,part,r['config']),r['weight']);y=d['y'][q]
            groups,inverse=np.unique(d['groups'][q],return_inverse=True)
            blocks=[np.flatnonzero(inverse==i) for i in range(len(groups))]
            rng=np.random.default_rng(880923+k+(100 if part=='valid' else 0));gains=[]
            for _ in range(1500):
                ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
                if metric in ('AUROC','AUPRC') and len(np.unique(y[ix]))<2:continue
                gain=v.r83.gain(metric,y[ix],base[ix],pred[ix])
                if np.isfinite(gain):gains.append(gain)
            record=dict(task=r['task'],part=part,metric=metric,rows=len(q),scaffolds=len(groups),
                point_gain=v.r83.gain(metric,y,base,pred),replicates=len(gains),
                percentile_95=np.quantile(gains,[.025,.975]).tolist(),positive_fraction=float(np.mean(np.array(gains)>0)))
            records.append(record);v.emit(stage='bootstrap_report_only',**record)
        old=np.array([float(row[4]) for row in original[1:] if row[0]==r['task']])
        new=np.array([float(row[4]) for row in actual[1:] if row[0]==r['task']])
        movements.append(dict(task=r['task'],rows=len(new),spearman_to_anchor=float(spearmanr(old,new).statistic),
            same_order=bool(np.array_equal(np.argsort(old,kind='stable'),np.argsort(new,kind='stable'))),
            quantile_new=np.quantile(new,[0,.1,.5,.9,1]).tolist()))
    v.b.save(v.OUT/'diagnostics.json',dict(status='COMPLETE',selection_effect='NONE_REPORT_ONLY',
        bootstrap=records,test_prediction_movement=movements,
        limitations='Historically reused labels, proxy baseline and conditional assay features. No estimate of official score or elimination of selection bias.'))


if __name__=='__main__':main()
