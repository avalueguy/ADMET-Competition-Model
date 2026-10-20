"""Reconstruct fitted-only features and replay all selected final models remotely."""
import json
import numpy as np
from catboost import CatBoostRegressor
import admet_v24_models as b
import admet_v51_run as v


def main():
    b.remote_only();d=b.build_data();c=v.chemistry(d)
    replay=json.loads((v.OUT/'fusion_replay.json').read_text())
    records=[]
    for task,r in replay.items():
        k=d['tasks'].index(task);rows=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2))
        x,prior,_,chosen=v.design(d,c,k,rows)
        probe=dict(d);probe['y']=d['y'].copy();probe['y'][q]=1e12
        xx,pp,_,cc=v.design(probe,c,k,rows)
        np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(prior,pp);np.testing.assert_array_equal(chosen,cc)
        dest=v.OUT/task/'final';npz=np.load(dest/'predictions.npz')
        np.testing.assert_array_equal(npz['fit'],rows);np.testing.assert_array_equal(npz['query'],q)
        pred=[]
        for family in range(3):
            parts=[];offset=prior if family else np.zeros_like(prior)
            for seed in v.P['seeds']:
                model=CatBoostRegressor();model.load_model(str(dest/f'expert{family}_{seed}.cbm'))
                pp=model.predict(x[d['mi'][q]])+offset[d['mi'][q]]
                single=model.predict(x[d['mi'][q[:1]]])+offset[d['mi'][q[:1]]]
                np.testing.assert_allclose(pp[:1],single,atol=1e-10,rtol=1e-10)
                parts.append(pp)
            pred.append(np.mean(parts,0))
        np.testing.assert_allclose(pred,npz['p'],atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(pred,r['source'],atol=1e-9,rtol=1e-9)
        records.append(dict(task=task,model_reload_passed=True,query_labels_invariant=True,
            singleton_prediction_invariant=True,preprocessor_fit_only=True))
        b.emit('v51_verified',task=task)
    b.save(v.OUT/'model_replay.json',dict(status='PASS',signature=v.signature(),tasks=records,
        local_gpu_training=False,test_labels_used=False))


if __name__=='__main__':main()
