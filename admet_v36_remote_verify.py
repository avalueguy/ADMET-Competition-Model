"""Independently reload the kernel and XGBoost sources selected in the final release."""
import json,joblib
import numpy as np
import torch
from xgboost import XGBClassifier,XGBRegressor
import admet_v24_models as b
from admet_v36_run import ROOT,OUT,SOURCES,signature
import admet_v33_run as n
import admet_v34_run as kmodel
import admet_v35_run as xmodel
def main():
    b.remote_only();torch.backends.cuda.matmul.allow_tf32=True;d=b.build_data();other=n.other_cache(d,[0,1]);records=[]
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    verified33=json.loads((n.OUT/'model_replay.json').read_text());assert verified33['status']=='FINAL_MODEL_REPLAY_PASSED' and verified33['signature']==n.signature()
    done=set()
    for r in report['tasks']:
        if r['selection'] is None:continue
        task=r['task'];k=d['tasks'].index(task);m=r['metric'];cls=m in ['AUROC','AUPRC']
        rows=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2))
        for source in r['selection']['sources']:
            v=source['version']
            if (task,v) in done:continue
            done.add((task,v));module=SOURCES[v];z=np.load(module.OUT/task/'final/predictions.npz')
            assert str(z['signature'])==module.signature();np.testing.assert_array_equal(z['fit'],rows);np.testing.assert_array_equal(z['query'],q)
            if v==33:
                assert task in {rr['task'] for rr in verified33['tasks']}
            elif v==34:
                kernels,tf,_=kmodel.matrices(d,k,rows,q,[0,1],other);parts=[]
                for j,(_,kernel) in enumerate(kernels):
                    ck=joblib.load(module.OUT/task/'final'/f'kernel{j}.joblib');assert ck['signature']==module.signature();values=[]
                    for model in ck['models']:
                        p=model.decision_function(kernel.astype(np.float64)) if cls else model.predict(kernel.astype(np.float64))
                        values.append(__import__('scipy').special.expit(p) if cls else (p if m=='Spearman' else ck['target'].decode(p)))
                    parts.append(np.mean(values,0))
                np.testing.assert_allclose(np.stack(parts),z['p'],rtol=1e-8,atol=1e-8)
            else:
                x,_,_,tf,_=n.features(d,k,rows,q,[0,1],other);parts=[];klass=XGBClassifier if cls else XGBRegressor
                for family in range(2):
                    values=[]
                    for seed in xmodel.P['seeds']:
                        model=klass();model.load_model(module.OUT/task/'final'/f'boost{family}_{seed}.ubj')
                        p=model.predict_proba(x['query']['trees'][family])[:,1] if cls else model.predict(x['query']['trees'][family])
                        values.append(p if cls or m=='Spearman' else tf.decode(p))
                    parts.append(np.mean(values,0))
                np.testing.assert_allclose(np.stack(parts),z['p'],rtol=2e-5,atol=2e-5)
            records.append(dict(task=task,version=v,model_reload_passed=True));b.emit('v36_replay',task=task,version=v,passed=True)
    b.save(OUT/'model_replay.json',dict(status='FINAL_MODEL_REPLAY_PASSED',signature=signature(),sources=records,local_gpu_training=False))
if __name__=='__main__':main()
