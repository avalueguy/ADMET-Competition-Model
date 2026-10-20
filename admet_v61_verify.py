"""Independent final-kernel model reload, complete feature and CSV replay."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import json
import numpy as np,torch,joblib
import admet_v24_models as b
import admet_v27_models as graph
import admet_v61_run as v

def main():
    b.remote_only();torch.backends.cuda.matmul.allow_tf32=False;d=b.build_data();g=graph.graphs(d);views=v.chemistry(d);replay=json.loads((v.OUT/'fusion_replay.json').read_text());checks=[]
    for task,r in replay.items():
        k=d['tasks'].index(task);m=d['metrics'][k];s=r['selection'];j=s['variant'];family=v.P['configurations'][j][0];rr=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2));dest=v.OUT/task/'final';z=np.load(dest/'predictions.npz');saved=joblib.load(dest/f'config{j}.joblib')
        assert saved['signature']==v.signature();np.testing.assert_array_equal(saved['rows'],rr);np.testing.assert_array_equal(z['fit'],rr);np.testing.assert_array_equal(z['query'],q)
        matrices,pre,audit=v.matrices(d,g,views,k,rr,q,'final',family==0);model=saved['model'];x=matrices[family][1];cls=m in ['AUROC','AUPRC'];pp=model.decision_function(x) if cls else model.predict(x)
        one=model.decision_function(x[:1]) if cls else model.predict(x[:1]);np.testing.assert_allclose(pp[:1],one,atol=1e-9,rtol=1e-9)
        if m=='MAE':pp=saved['target'].decode(pp)
        np.testing.assert_allclose(pp,z['p'][j],atol=1e-9,rtol=1e-9);np.testing.assert_allclose(pp,r['branch'],atol=1e-9,rtol=1e-9);merged=b.blend(m,np.array(r['baseline']),pp,s['weight'])
        if task=='ppbr_az':merged=merged.clip(0,100)
        np.testing.assert_allclose(merged,r['result'],atol=1e-9,rtol=1e-9);checks.append(dict(task=task,model_reload_passed=True,batch_invariance=True,fit_query_indices_exact=True));b.emit('v61_model_replay',**checks[-1])
    b.save(v.OUT/'model_replay.json',dict(status='PASS',tasks=checks,test_labels_used=False,local_gpu_training=False))

if __name__=='__main__':main()
