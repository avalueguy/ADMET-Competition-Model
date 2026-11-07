"""Fit-only feasibility for marginalizing measured-context replicates.

Uses only original v145 v127 checkpoints and permitted other-assay training
observations. No target labels, outer query predictions, or scores are read.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='4'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v127_low_classification as component

TASKS=('herg','cyp3a4_substrate_carbonmangels')


def tables(d,k,part):
    original,cols,source,_,_=component.v.old.assays.assays(d,k,part)
    assert np.all(d['tid'][source]!=k) and k not in cols
    lookups=[{},{}]
    for row in source:
        for j,keys in enumerate((d['canonical'],d['neutral_identity'])):
            lookups[j].setdefault((str(keys[row]),int(d['tid'][row])),[]).append(float(d['y'][row]))
    for lookup in lookups:
        for key,values in lookup.items():
            lookup[key]=np.sort(np.asarray(values,dtype=float),kind='stable')
    return original,cols,source,lookups


def samples(d,k,part,rows,physical):
    old,cols,source,lookup=tables(d,k,part)
    n=len(cols)
    assert n==21 and old.shape[1]==5*n
    context=np.repeat(old[rows][None],16,axis=0)
    uniforms=qmc.Sobol(d=2*n,scramble=False).random_base2(4)
    changed=np.zeros(len(rows),dtype=bool)
    for i,row in enumerate(rows):
        for block,keys in enumerate((d['canonical'],d['neutral_identity'])):
            for j,t in enumerate(cols):
                values=lookup[block].get((str(keys[row]),int(t)))
                if values is None: continue
                assert np.float32(np.median(values))==old[row,2*block*n+j]
                if len(values)>1 and np.ptp(values)>0:
                    changed[i]=True
                    context[:,i,2*block*n+j]=values[np.minimum((uniforms[:,block*n+j]*len(values)).astype(int),len(values)-1)]
    return [np.c_[a,physical].astype(np.float32) for a in context],changed,source


def main():
    dest=ROOT/'docs/v294_context_uncertainty_audit.json'
    assert not dest.exists(),'Diagnostic is one-shot; inspect existing report'
    d=v.load();records=[]
    for task in TASKS:
        k=d['tasks'].index(task);parts=[]
        for part in ('fold0','fold1'):
            fit,q,x,_,source=component.v.features(d,k,part,'physchem')
            matrices,changed,source2=samples(d,k,part,fit,x[:,105:])
            np.testing.assert_array_equal(source,source2)
            assert all(a.shape==x.shape for a in matrices)
            means=[];original=[]
            for seed in component.SEEDS:
                model=CatBoostClassifier().load_model(str(component.OUT/'models'/task/part/'physchem'/f'model_{seed}.cbm'))
                original.append(model.predict(x,prediction_type='RawFormulaVal',thread_count=1))
                means.append(np.mean([model.predict(a,prediction_type='RawFormulaVal',thread_count=1) for a in matrices],axis=0))
            delta=np.mean(means,axis=0)-np.mean(original,axis=0)
            parts.append(dict(part=part,fit_rows=len(fit),affected_rows=int(changed.sum()),
                affected_fraction=float(changed.mean()),changed_logit_rows=int((np.abs(delta)>1e-12).sum()),
                delta_absolute_quantiles=np.quantile(np.abs(delta),[0,.5,.9,1]).tolist(),
                supported=bool(changed.sum()>=10 and changed.mean()>=.05 and np.max(np.abs(delta))>1e-6)))
        records.append(dict(task=task,parts=parts,supported=all(r['supported'] for r in parts)))
    report=dict(status='FIT_ONLY_FEASIBILITY_COMPLETE',at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=v.b.sha(__file__),tasks_predeclared=TASKS,records=records,
        target_labels_read=False,query_predictions_computed=False,scores_computed=False,
        old_model_changes=False,new_models_trained=False,
        hypothesis='Original component inputs collapse repeated other-assay observations to medians. For nonlinear trees E[f(context)] differs from f(median(context)). Integrate empirical replicate values while preserving existing identities/masks/spreads/physical inputs and immutable checkpoints.',
        qualification='Fit-only context conflicts and prediction sensitivity justify feasibility only, not generalization or hidden gain.',
        historical_overlap={'v264':'Noise model on own-target repeated labels and newGP fitting. Here only input uncertainty of other-assay measurements, fixed existing v145 checkpoints.',
            'v271':'Paired-task residual/innovation covariance with newreference models. Here no target fitting, innovation covariance or proxy offsets.',
            'v284_v289':'Stereochemical identity replacement/newtrees. Here original identities and trees retained; repeated measurement marginalization only.',
            'v94':'Assay-dropout neural training. Here no dropout, imputation, newnet, label fitting or parameter sweep.'},
        protocol_required_before_scores=True)
    v.b.save(dest,report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
