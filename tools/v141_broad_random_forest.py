"""Randomized molecular forests to complement existing boosting/network ensembles."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import hashlib,json,sys,time
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v131_current_controls as v
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v141';WEIGHTS=(.35,.75)
CONFIGS=(dict(view='augmented',leaf=3,features=.35),dict(view='full',leaf=2,features=.08))
def sig():return hashlib.sha256((b.sha(__file__)+v.sig()).encode()).hexdigest()
def key(c):return c['view']+'_leaf'+str(c['leaf'])
def design(d,k,part,c):
    fit,q,x,qx,source=v.features(d,k,part,c['view']);median=np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0. for z in x.T],np.float32)
    return fit,q,np.where(np.isfinite(x),x,median),np.where(np.isfinite(qx),qx,median),source,median
def predict(d,k,part,c):
    folder=OUT/'models'/d['tasks'][k]/part/key(c);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);return z['p'].copy()
    fit,q,x,qx,source,median=design(d,k,part,c);y=d['y'][fit];m=d['metrics'][k];cls=m in ('AUROC','AUPRC')
    target=r83.ranknorm(y) if m=='Spearman' else y
    model=(ExtraTreesClassifier if cls else ExtraTreesRegressor)(n_estimators=350,max_depth=18,min_samples_leaf=c['leaf'],max_features=c['features'],
        random_state=1414026,n_jobs=4,bootstrap=False)
    folder.mkdir(parents=True,exist_ok=True);start=time.monotonic();model.fit(x,target)
    p=model.predict_proba(qx)[:,1] if cls else model.predict(qx);mp=folder/'model.joblib';joblib.dump(model,mp,compress=3)
    replay=joblib.load(mp);rp=replay.predict_proba(qx)[:,1] if cls else replay.predict(qx);np.testing.assert_allclose(p,rp,atol=1e-12,rtol=0)
    np.savez_compressed(file,signature=sig(),fit=fit,query=q,p=p,source=source,median=median)
    print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,config=c,seconds=round(time.monotonic()-start))),flush=True);return p
def record(d,k,part,c,w):
    q=v.split(d,k,part)[1];m=d['metrics'][k];a=v.baseline(d,k,part);z=predict(d,k,part,c);p=v.mix(m,a,z,w)
    return dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],a),expert=r83.score(m,d['y'][q],z),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],a,p))
def main():
    d=v.load();OUT.mkdir(exist_ok=True);doc=dict(signature=sig(),tasks=v.tasks(d),configs=CONFIGS,weights=WEIGHTS,
        model='350 bounded-depth ExtraTrees; augmented chemical roles/shape or 12288 fragment counts, with target-excluded measurements. Fit-only median imputation.',
        selection='Max minimum fold0/1 gain, frozen before fold2/valid; all4 positive and mean exceeds threshold; no fallback.',
        baseline='v130 reconstructed/proxy controls; labels reused; official improvement unconfirmed',external_data=False,test_labels_used=False)
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(doc))
    else:b.save(path,doc)
    development=[];selected=[]
    for task in v.tasks(d):
        k=d['tasks'].index(task);choices=[]
        for c in CONFIGS:
            for w in WEIGHTS:
                parts=[record(d,k,part,c,w) for part in ('fold0','fold1')]
                choices.append(dict(task=task,metric=d['metrics'][k],config=c,weight=w,parts=parts,objective=min(r['gain'] for r in parts)))
        best=max(choices,key=lambda r:r['objective']);development.extend(choices)
        if best['objective']>v.threshold(d['metrics'][k]):selected.append(best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=development));print(json.dumps(dict(stage='development',best=best)),flush=True)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=development));b.save(OUT/'frozen_selection.json',dict(signature=sig(),selected=selected))
    records=[]
    for chosen in selected:
        k=d['tasks'].index(chosen['task']);parts=[record(d,k,p,chosen['config'],chosen['weight']) for p in ('fold0','fold1','fold2','valid')]
        accepted=bool(min(r['gain'] for r in parts)>0 and np.mean([r['gain'] for r in parts])>v.threshold(d['metrics'][k]))
        records.append(dict(**chosen,confirmation=parts,accepted=accepted));print(json.dumps(dict(stage='confirmation',**records[-1])),flush=True)
        if accepted:predict(d,k,'final',chosen['config'])
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']]))
if __name__=='__main__':main()
