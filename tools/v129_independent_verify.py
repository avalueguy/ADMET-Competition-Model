"""Reload accepted low-task experts and audit immutable v124 task replacement."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from scipy.special import expit
from catboost import CatBoostClassifier,CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v125_low_task_baseline as v
import v126_dual_target_clearance as dual
import v127_low_classification as tree
import v128_conditional_margin as kernel
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v129'

def reload(d,k,part,module,chosen):
    if module is kernel:
        fit,q,mm=kernel.matrices(d,k,part);x,qx=mm[chosen['config']['view']]
        folder=kernel.OUT/'models'/d['tasks'][k]/part;file=folder/(kernel.key(chosen['config'])+'.joblib')
        model=joblib.load(file);p=expit(model.decision_function(qx));hashes={file.name:b.sha(file)}
        # Prediction depends only on fitted state, and respects row permutation.
        reverse=expit(model.decision_function(qx[::-1]))[::-1];np.testing.assert_allclose(p,reverse,atol=1e-14,rtol=0)
        cache=np.load(folder/(kernel.key(chosen['config'])+'.npz'))
    else:
        fit,q,x,qx,source=v.features(d,k,part,chosen['view']);folder=module.OUT/'models'/d['tasks'][k]/part/chosen['view']
        cache=np.load(folder/'predictions.npz');members=[];hashes={}
        for seed in module.SEEDS:
            file=folder/f'model_{seed}.cbm';net=CatBoostClassifier() if module is tree else CatBoostRegressor();net.load_model(str(file));hashes[file.name]=b.sha(file)
            if module is tree:members.append(net.predict(qx,prediction_type='RawFormulaVal'))
            else:
                raw=net.predict(qx)
                if d['metrics'][k]=='Spearman':members.append(.5*r83.rankdata(raw[:,0])/len(q)+.5*r83.rankdata(raw[:,1])/len(q))
                else:members.append(raw[:,0]*cache['scale'][0]+cache['center'][0])
        p=expit(np.mean(members,axis=0)) if module is tree else np.mean(members,axis=0)
    np.testing.assert_array_equal(fit,cache['fit']);np.testing.assert_array_equal(q,cache['query']);assert str(cache['signature'])==module.sig()
    np.testing.assert_allclose(p,cache['p'],atol=1e-12,rtol=0)
    return fit,q,p,hashes

def main():
    inputs=json.loads((ROOT/'releases/v125_baseline_inputs.manifest.json').read_text())
    for row in inputs['files']:
        path=ROOT/row['path'];assert path.stat().st_size==row['bytes'] and b.sha(path)==row['sha256']
    d=v.load();rng=np.random.default_rng(1294026);checks=[];selected={};experiments=[]
    for name,module in [('v126',dual),('v127',tree),('v128',kernel)]:
        frozen=json.loads((module.OUT/'frozen_selection.json').read_text());conf=json.loads((module.OUT/'confirmation.json').read_text())
        assert conf['status']=='COMPLETE' and conf['signature']==frozen['signature']==module.sig()
        experiments.append(dict(version=name,accepted=conf['accepted'],confirmation_sha256=b.sha(module.OUT/'confirmation.json')))
        for r in conf['records']:
            if not r['accepted']:continue
            assert any(r['task']==x['task'] and r['parts']==x['parts'] for x in frozen['selected'])
            assert r['task'] in v.TASKS
            objective=min(x['gain'] for x in r['parts'])
            if r['task'] not in selected or objective>selected[r['task']]['objective']:
                selected[r['task']]=dict(version=name,record=r,objective=objective,weight=module.WEIGHT)
    for task,choice in selected.items():
        module={'v126':dual,'v127':tree,'v128':kernel}[choice['version']];k=d['tasks'].index(task)
        for part in ('fold0','fold1','fold2','valid','final'):
            fit,q,p,hashes=reload(d,k,part,module,choice['record']);a=v.baseline(d,k,part);candidate=v.mix(d['metrics'][k],a,p,module.WEIGHT)
            record=dict(task=task,version=choice['version'],part=part,fit=len(fit),query=len(q),model_hashes=hashes,model_reload=True)
            if part!='final':
                actual=r83.score(d['metrics'][k],d['y'][q],candidate);expected=next(x for x in choice['record']['confirmation'] if x['part']==part)
                np.testing.assert_allclose(actual,expected['candidate'],atol=1e-12,rtol=0);record.update(expected)
                if part in ('fold2','valid'):
                    groups=np.unique(d['groups'][q]);deltas=[]
                    for _ in range(1000):
                        drawn=rng.choice(groups,len(groups),replace=True);ix=np.concatenate([np.flatnonzero(d['groups'][q]==g) for g in drawn])
                        if len(np.unique(d['y'][q][ix]))<2:continue
                        value=r83.gain(d['metrics'][k],d['y'][q][ix],a[ix],candidate[ix])
                        if np.isfinite(value):deltas.append(value)
                    record.update(bootstrap_samples=len(deltas),scaffold_bootstrap_gain_ci95=np.quantile(deltas,[.025,.975]).tolist())
            # Query-label perturbation must not change training/query feature design.
            before=v.features(d,k,part,'full');saved=d['y'][q].copy();d['y'][q]=987654;d.pop('_v96_assays',None)
            after=v.features(d,k,part,'full');d['y'][q]=saved;d.pop('_v96_assays',None)
            for x,z in zip(before,after):np.testing.assert_array_equal(x,z)
            if module is kernel:
                _,_,original=kernel.matrices(d,k,part);d['y'][q]=123456;d.pop('_v96_assays',None);_,_,perturbed=kernel.matrices(d,k,part)
                d['y'][q]=saved;d.pop('_v96_assays',None)
                for key in original:
                    for x,z in zip(original[key],perturbed[key]):np.testing.assert_array_equal(x,z)
            record['query_label_perturbation']=True;checks.append(record)
    report=dict(status='PASS',baseline='v124',baseline_sha256=v.ANCHOR_SHA,matched_baseline_input_hashes_verified=True,selected=selected,checks=checks,experiments=experiments,
        protected_tasks_immutable=[t for t in d['tasks'] if t not in selected],test_labels_used=False,
        limitations='Development labels previously reused. Matched CYP3A4 branch, approximate upstream history for other tasks. Bootstrap diagnostics do not remove reuse bias. Official gain unconfirmed.')
    b.save(OUT/'independent_verification.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
