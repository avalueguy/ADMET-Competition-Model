"""Rebuild authorized other-assay features and reload final foundation experts."""
import os,sys
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'admet_runtime/v63_packages'))
import json,gc
import numpy as np,torch,joblib
from tabicl import TabICLClassifier,TabICLRegressor
import admet_v24_models as b
import admet_v65_run as v

def main():
    b.remote_only();torch.backends.cuda.matmul.allow_tf32=False;d=b.build_data();replay=json.loads((v.OUT/'fusion_replay.json').read_text());checks=[]
    for task,r in replay.items():
        k=d['tasks'].index(task);m=d['metrics'][k];s=r['selection'];j=s['variant'];rr=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2));dest=v.OUT/task/'final';z=np.load(dest/'predictions.npz');saved=joblib.load(dest/'preprocessing.joblib');assert saved['signature']==v.signature();np.testing.assert_array_equal(z['fit'],rr);np.testing.assert_array_equal(z['query'],q);xx,_,audit=v.design(d,k,rr,q,[0,1],[j]);cls=m in ['AUROC','AUPRC'];model=(TabICLClassifier if cls else TabICLRegressor).load(dest/f'family{j}.pt',device='cuda');p=model.predict_proba(xx['query'][j])[:,1] if cls else model.predict(xx['query'][j],output_type='median' if m=='MAE' else 'mean')
        v.stable.check(model)
        if m=='MAE':p=saved['target'].decode(p)
        np.testing.assert_allclose(p,z['p'][j],atol=2e-5,rtol=2e-5);np.testing.assert_allclose(p,r['branch'],atol=2e-5,rtol=2e-5);merged=b.blend(m,np.array(r['baseline']),p,s['weight'])
        if task=='ppbr_az':merged=merged.clip(0,100)
        np.testing.assert_allclose(merged,r['result'],atol=2e-5,rtol=2e-5);checks.append(dict(task=task,model_reload_passed=True,fit_query_indices_exact=True,own_task_assay_excluded=True,conditional_features=True));b.emit('v65_model_replay',**checks[-1]);del model;gc.collect();torch.cuda.empty_cache()
    b.save(v.OUT/'model_replay.json',dict(status='PASS',tasks=checks,test_labels_used=False,local_gpu_training=False))

if __name__=='__main__':main()
