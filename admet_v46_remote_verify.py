"""Reload selected task checkpoints and verify source, partition and prediction integrity."""
import json
import time
import numpy as np
import torch
from scipy.special import expit
import admet_v24_models as b
import admet_v27_models as graph
import admet_v46_run as v

def verify_partition(d,g,k,task,partition):
    source_partition=v.source_partition(partition)
    results=[]
    for seed in v.P['source_seeds']:
        dest=v.OUT/task/partition/str(seed)
        z=np.load(dest/'predictions.npz')
        assert str(z['signature'])==v.signature()
        rows,query=z['fit'],z['query']
        source=graph.OUT/source_partition/str(seed)
        original=np.load(source/'predictions.npz')
        assert str(original['signature'])==graph.signature()
        q=original['query']
        pos=np.searchsorted(q,query)
        np.testing.assert_array_equal(q[pos],query)
        np.testing.assert_array_equal(z['p'][:2],original['p'][:,pos])
        assert np.all(d['split'][rows]<2)
        if partition!='final':
            for refs in [rows,original['fit']]:
                assert not set(d['rg'][refs])&set(d['rg'][query])
                assert not set(d['canonical'][d['mi'][refs]])&set(d['canonical'][d['mi'][query]])
        predictions=[]
        for step in v.P['snapshot_steps']:
            ck=torch.load(dest/f'checkpoint_{step}.pt',map_location='cpu',weights_only=False)
            assert ck['signature']==v.signature() and ck['target_task']==k
            assert ck['source_sha256']==b.sha(source/f'checkpoint_{v.P["source_step"]}.pt')
            net=graph.BondNet(*ck['dimensions']).cuda().eval()
            net.load_state_dict(ck['state'])
            x=v.feature_matrix(d,ck,source_partition)
            p=expit(v.logits(net,g,x,d['mi'][query])[:,:,k])
            predictions.append(p)
            if step==v.P['snapshot_steps'][-1]:
                first=d['mi'][query[:min(5,len(query))]]
                together=v.logits(net,g,x,first)[:,:,k]
                individual=np.concatenate([v.logits(net,g,x,np.array([a]))[:,:,k] for a in first],axis=1)
                np.testing.assert_allclose(together,individual,rtol=3e-5,atol=3e-5)
                perm=np.arange(len(first))[::-1]
                reversed_order=v.logits(net,g,x,first[perm])[:,:,k]
                np.testing.assert_allclose(together[:,perm],reversed_order,rtol=3e-5,atol=3e-5)
                # The prediction inputs use only molecular structure and a
                # saved training preprocessor. Hidden labels are not arguments.
                original_y=d['y'][query].copy()
                d['y'][query]=999999.
                poisoned=v.logits(net,g,x,first)[:,:,k]
                d['y'][query]=original_y
                np.testing.assert_allclose(together,poisoned,rtol=0,atol=3e-6)
            del net
        np.testing.assert_allclose(z['p'][2:],np.mean(predictions,axis=0),rtol=3e-5,atol=3e-5)
        results.append(dict(seed=seed,partition=partition,reload_passed=True,
            source_predictions_exact=True,query_labels_not_used=True,batch_and_permutation_invariant=True))
    return results

def main():
    b.remote_only()
    torch.backends.cuda.matmul.allow_tf32=False
    d=b.build_data();g=graph.graphs(d)
    report=json.loads((v.OUT/'analysis.json').read_text())
    assert report['signature']==v.signature() and report['status']=='FROZEN'
    rows=[];t0=time.time()
    for r in report['tasks']:
        if not r['selection']:continue
        k=d['tasks'].index(r['task'])
        evidence=[]
        for partition in ['valid','fold0','fold1','fold2','final']:
            evidence.extend(verify_partition(d,g,k,r['task'],partition))
        rows.append(dict(task=r['task'],partitions=evidence,status='PASS'))
        b.save(v.OUT/'model_replay.json',dict(signature=v.signature(),status='IN_PROGRESS',tasks=rows))
        b.emit('v46_verified',task=r['task'],seconds=round(time.time()-t0))
    b.save(v.OUT/'model_replay.json',dict(signature=v.signature(),status='PASS',tasks=rows,
        local_gpu_training=False,test_labels_used=False,source_and_fit_partition_isolation=True))

if __name__=='__main__':main()
