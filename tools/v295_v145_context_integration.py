"""Empirical-context integration in immutable v145 component checkpoints.

This optimizes inference of actual v145 models, without fitting a new model,
changing trees, seeds, identities, labels, or using rejected expert outputs.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='4'
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
import v127_low_classification as original
import v294_context_uncertainty_audit as audit

b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v295'
TASKS=('herg','cyp3a4_substrate_carbonmangels')
CONFIGS=(dict(integration_points=16,rule='unscrambled_sobol_empirical_marginals'),)
SEEDS=original.SEEDS
METHOD=('Integrate existing actualv145v127physchem checkpoints over empirical permitted '
 'other-assay replicate measurements instead of a single median.16 deterministic Sobol '
 'product-marginal inputstates; original identities, masks, neutral spreads and physical '
 'descriptors unchanged. Mean rawlogits over16states and2originalmodels then sigmoid. '
 'Original.5 component rankblend, originalv148.2/.45 taskblend and gates. '
 'No newmodels/trees/seeds/training, no label/prediction offsets, no failedexpert reuse. '
 'Fit-only contextreplicate audit supports2tasks;20protected endpoints,19runnercontrols/HIAexcluded.')


def sig():
    packages='|'.join(importlib.metadata.version(p) for p in ('numpy','scipy','catboost'))
    return hashlib.sha256((b.sha(__file__)+b.sha(audit.__file__)+v.sig()+original.sig()+b.sha(runner.__file__)+packages).encode()).hexdigest()


def load():
    threadpool_limits(limits=4)
    return v.load()


def key(c):
    assert c==CONFIGS[0]
    return 'v145_context_integral16'


def same(a,z):
    assert set(a)==set(z)
    for name in a:
        np.testing.assert_array_equal(a[name],z[name])


def design(d,k,part,c=None):
    assert d['tasks'][k] in TASKS
    fit,q,x,qx,source=original.v.features(d,k,part,'physchem')
    old,cols,old_source,_,_=original.v.old.assays.assays(d,k,part)
    assert len(cols)==21 and old.shape[1]==105 and x.shape[1]==qx.shape[1]==540
    np.testing.assert_array_equal(source,old_source)
    np.testing.assert_array_equal(x[:,:105],old[fit])
    np.testing.assert_array_equal(qx[:,:105],old[q])
    matrices,changed,new_source=audit.samples(d,k,part,q,qx[:,105:])
    np.testing.assert_array_equal(source,new_source)
    stack=np.stack(matrices)
    assert stack.shape==(16,len(q),540)
    for a in stack:
        # Only measured values vary: masks, neutralspread, and physicalsuffix exact.
        for first,last in ((21,42),(63,105),(105,540)):
            np.testing.assert_array_equal(a[:,first:last],qx[:,first:last])
    return dict(fit=fit,query=q,source=source,x=x,qx=qx,states=stack,affected=changed)


def original_replay(d,k,part):
    fit,q,x,qx,source=original.v.features(d,k,part,'physchem')
    members=[];hashes={}
    for seed in SEEDS:
        path=original.OUT/'models'/d['tasks'][k]/part/'physchem'/f'model_{seed}.cbm'
        assert path.exists(),'No implicit original model training'
        net=CatBoostClassifier().load_model(str(path))
        assert net.tree_count_==900
        members.append(net.predict(qx,prediction_type='RawFormulaVal',thread_count=1))
        hashes[path.relative_to(ROOT).as_posix()]=b.sha(path)
    p=expit(np.mean(members,axis=0))
    z=np.load(path.parent/'predictions.npz')
    assert str(z['signature'])==original.sig()
    np.testing.assert_array_equal(z['fit'],fit)
    np.testing.assert_array_equal(z['query'],q)
    np.testing.assert_allclose(p,z['p'],atol=1e-12,rtol=0)
    carried=original.v.mix('AUROC',original.v.baseline(d,k,part),p,.5)
    np.testing.assert_allclose(carried,v.baseline(d,k,part),atol=1e-12,rtol=0)
    return hashes


def infer(d,k,part,a):
    members=[];hashes={}
    for seed in SEEDS:
        path=original.OUT/'models'/d['tasks'][k]/part/'physchem'/f'model_{seed}.cbm'
        net=CatBoostClassifier().load_model(str(path))
        assert net.tree_count_==900
        # Keep every state separate to make summation deterministic under permutations.
        values=np.stack([net.predict(x,prediction_type='RawFormulaVal',thread_count=1) for x in a['states']])
        members.append(np.mean(values,axis=0))
        hashes[path.relative_to(ROOT).as_posix()]=b.sha(path)
    component=expit(np.mean(members,axis=0))
    p=original.v.mix('AUROC',original.v.baseline(d,k,part),component,.5)
    assert np.isfinite(p).all()
    return p,hashes


def replay(d,k,part,c):
    a=design(d,k,part,c)
    z=np.load(OUT/'models'/d['tasks'][k]/part/key(c)/'predictions.npz')
    assert str(z['signature'])==sig()
    for name in a:
        np.testing.assert_array_equal(a[name],z[name])
    p,hashes=infer(d,k,part,a)
    np.testing.assert_array_equal(p,z['p'])
    return p,hashes


def predict(d,k,part,c):
    if d['tasks'][k] not in TASKS:
        return v.baseline(d,k,part)
    folder=OUT/'models'/d['tasks'][k]/part/key(c)
    file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file)
        assert str(z['signature'])==sig()
        np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1])
        return z['p'].copy()
    assert not folder.exists(),'Inspect partial inference artifact; no silent retry'
    started=time.monotonic();a=design(d,k,part,c)
    original_hashes=original_replay(d,k,part)
    p,hashes=infer(d,k,part,a)
    assert hashes==original_hashes
    folder.mkdir(parents=True)
    np.savez_compressed(file,signature=sig(),p=p,**a)
    b.save(folder/'metadata.json',dict(signature=sig(),original_model_hashes=hashes,
        new_models_trained=0,original_component_weight=.5,integration_points=16,
        independent_empirical_marginals=True,target_query_labels_used=False,seconds=time.monotonic()-started))
    print(json.dumps(dict(stage='integrate_immutable_v145',task=d['tasks'][k],part=part,
        query_rows=len(a['query']),new_models_trained=0,seconds=round(time.monotonic()-started))),flush=True)
    return p


def independent_input_check(d):
    # Force fresh allowed-label tables, no cached integration state.
    d.pop('_v96_assays',None)
    for task in TASKS:
        k=d['tasks'].index(task)
        for part in v.PARTS:
            design(d,k,part)


def smoke(d):
    from tempfile import TemporaryDirectory
    checks=[]
    for task in TASKS:
        k=d['tasks'].index(task)
        # Fit rows only: smoke must not open query prediction scores before freeze.
        fit,q,x,qx,source=original.v.features(d,k,'fold0','physchem')
        matrices,changed,source2=audit.samples(d,k,'fold0',fit,x[:,105:])
        assert all(a.shape==x.shape for a in matrices)
        saved=d['y'].copy()
        try:
            mask=(d['tid']==k)&~np.isin(np.arange(len(d['y'])),fit)
            d['y'][mask]=999999.
            d.pop('_v96_assays',None)
            alternative,changed2,source3=audit.samples(d,k,'fold0',fit,x[:,105:])
            np.testing.assert_array_equal(np.stack(matrices),np.stack(alternative))
            np.testing.assert_array_equal(source2,source3)
            np.testing.assert_array_equal(changed,changed2)
        finally:
            d['y'][:]=saved;d.pop('_v96_assays',None)
        path=original.OUT/'models'/task/'fold0'/'physchem'/f'model_{SEEDS[0]}.cbm'
        model=CatBoostClassifier().load_model(str(path))
        expected=np.mean([model.predict(a,prediction_type='RawFormulaVal',thread_count=1) for a in matrices],axis=0)
        with TemporaryDirectory(dir=OUT) as tmp:
            copy=Path(tmp)/'model.cbm';model.save_model(str(copy))
            reload=CatBoostClassifier().load_model(str(copy))
            restored=np.mean([reload.predict(a,prediction_type='RawFormulaVal',thread_count=1) for a in matrices],axis=0)
            np.testing.assert_array_equal(expected,restored)
        reverse,_,_=audit.samples(d,k,'fold0',fit[::-1],x[::-1,105:])
        reordered=np.mean([model.predict(a,prediction_type='RawFormulaVal',thread_count=1) for a in reverse],axis=0)
        np.testing.assert_array_equal(expected,reordered[::-1])
        checks.append(dict(task=task,actual_original_checkpoint_reload=True,empirical_input_rebuild=True,
            row_order=True,nonfit_target_label_isolation=True,features=540,models_trained=0))
    # Independent analytical reference: all empirical quantiles sampled equally
    # for binary replicate sets, constant distributions identical to the input.
    from scipy.stats import qmc
    points=qmc.Sobol(d=42,scramble=False).random_base2(4)
    assert np.all((points<.5).sum(0)==8)
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),target_nonfit_label_isolation=True,
        deterministic_empirical_reference=True,independent_model_reload=True,checks=checks,
        scores_computed=False,query_predictions_computed=False,new_models_trained=0))
    print('SMOKE_PASS',flush=True)


def main():
    OUT.mkdir(exist_ok=True);d=load()
    if '--smoke' in sys.argv:
        smoke(d)
    else:
        assert json.loads((OUT/'smoke.json').read_text())['signature']==sig()
        runner.run(sys.modules[__name__],d)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        b.save(OUT/'failure.json',dict(status='FAILED',traceback=traceback.format_exc(),retry_permitted=False))
        raise
