"""Low-score classifiers on carried task controls, high-score outputs locked."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.special import expit
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v125_low_task_baseline as v
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v127';SEEDS=(1274026,1274027);WEIGHT=.5
TASKS=('cyp3a4_substrate_carbonmangels','herg','cyp2d6_substrate_carbonmangels','pgp_broccatelli')
def sig():return hashlib.sha256((b.sha(__file__)+v.sig()).encode()).hexdigest()
def predict(d,k,part,view):
    folder=OUT/'models'/d['tasks'][k]/part/view;file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);return z['p'].copy()
    fit,q,x,qx,source=v.features(d,k,part,view);y=d['y'][fit];folder.mkdir(parents=True,exist_ok=True)
    members=[];started=time.monotonic()
    for seed in SEEDS:
        net=CatBoostClassifier(loss_function='Logloss',iterations=900,depth=4,learning_rate=.025,l2_leaf_reg=30,
            random_seed=seed,thread_count=3,verbose=False,allow_writing_files=False,bootstrap_type='Bernoulli',subsample=.85,
            border_count=64,task_type='GPU',devices='0')
        net.fit(x,y);raw=net.predict(qx,prediction_type='RawFormulaVal');mp=folder/f'model_{seed}.cbm';net.save_model(str(mp))
        replay=CatBoostClassifier();replay.load_model(str(mp));np.testing.assert_allclose(raw,replay.predict(qx,prediction_type='RawFormulaVal'),atol=1e-10,rtol=0)
        members.append(raw)
    p=expit(np.mean(members,axis=0));np.savez_compressed(file,signature=sig(),fit=fit,query=q,source=source,p=p)
    print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,view=view,fit=len(fit),query=len(q),seconds=round(time.monotonic()-started))),flush=True);return p
def record(d,k,part,view):
    q=v.split(d,k,part)[1];a=v.baseline(d,k,part);expert=predict(d,k,part,view);p=v.mix(d['metrics'][k],a,expert,WEIGHT)
    return dict(part=part,rows=len(q),baseline=r83.score(d['metrics'][k],d['y'][q],a),expert=r83.score(d['metrics'][k],d['y'][q],expert),candidate=r83.score(d['metrics'][k],d['y'][q],p),gain=r83.gain(d['metrics'][k],d['y'][q],a,p))
def main():
    d=v.load();OUT.mkdir(exist_ok=True);doc=dict(signature=sig(),tasks=TASKS,seeds=SEEDS,weight=WEIGHT,views=['physchem','full'],
        model='GPU CatBoost 900 depth4; two seeds, mean logits. Official target-excluded context plus descriptors/full fingerprints.',
        selection='fold0/1 max-min view gain>.0025, frozen before other parts',confirmation='All4 gains>0, mean>.0025; no fallback.',
        baseline='Actual carried branch replay, exact v67 CYP3A4 target fit rows; upstream proxies for other endpoints.',protected_outputs_immutable=True,test_labels_used=False,external_data=False)
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(doc))
    else:b.save(p,doc)
    development=[];selected=[]
    for task in TASKS:
        k=d['tasks'].index(task);choices=[]
        # Explicitly perturb query labels before model fitting.
        for part in ('fold0','valid','final'):
            a=v.features(d,k,part,'full');q=a[1];saved=d['y'][q].copy();d['y'][q]=987654;d.pop('_v96_assays',None)
            other=v.features(d,k,part,'full');d['y'][q]=saved;d.pop('_v96_assays',None)
            for x,z in zip(a,other):np.testing.assert_array_equal(x,z)
        for view in ('physchem','full'):
            parts=[record(d,k,part,view) for part in ('fold0','fold1')];choices.append(dict(task=task,view=view,parts=parts,objective=min(x['gain'] for x in parts)))
        best=max(choices,key=lambda r:r['objective']);development.extend(choices)
        if best['objective']>.0025:selected.append(best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',records=development));print(json.dumps(dict(stage='development',best=best)),flush=True)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=development))
    b.save(OUT/'frozen_selection.json',dict(signature=sig(),selected=selected))
    records=[]
    for chosen in selected:
        k=d['tasks'].index(chosen['task']);parts=[record(d,k,part,chosen['view']) for part in ('fold0','fold1','fold2','valid')]
        accepted=bool(min(x['gain'] for x in parts)>0 and np.mean([x['gain'] for x in parts])>.0025)
        records.append(dict(**chosen,confirmation=parts,accepted=accepted));print(json.dumps(dict(stage='confirmation',**records[-1])),flush=True)
        if accepted:predict(d,k,'final',chosen['view'])
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],query_label_perturbation=True))
if __name__=='__main__':main()
