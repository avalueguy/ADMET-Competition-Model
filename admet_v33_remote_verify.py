"""Replay final models and prove held target labels cannot enter features."""
import json
import numpy as np
import torch
from catboost import CatBoostClassifier,CatBoostRegressor
import admet_v24_models as b
from admet_v33_run import ROOT,OUT,P,signature,features,other_cache,RetrievalNet,retrieve

def audit(d,other):
    records=[]
    for t in ['cyp2c9_substrate_carbonmangels','ppbr_az']:
        k=d['tasks'].index(t);rows=np.flatnonzero((d['tid']==k)&(d['split']==0))
        query=np.flatnonzero((d['tid']==k)&(d['split']==1))[:9]
        x,*_=features(d,k,rows,query,[0],other)
        perturbed=dict(d);perturbed['y']=d['y'].copy()
        active=(d['tid']==k)&~np.isin(np.arange(len(d['y'])),rows)
        perturbed['y'][active]=999999.;perturbed['y'][d['split']==2]=-888888.
        modified=other.copy();modified[:,k]=777777.
        xx,*_=features(perturbed,k,rows,query,[0],modified)
        for part in ['fit','query']:
            for name in ['fp','dense','aux']:
                np.testing.assert_array_equal(x[part][name],xx[part][name])
        records.append(dict(task=t,held_target_and_test_labels_invariant=True,target_neighbor_block_removed=True))
    # Retrieval must remove self and all same-scaffold identities even for fit molecules.
    refs=np.array([0]);q=refs.copy();values=np.array([12345.],np.float32)
    result=retrieve(d,q,refs,values,True)
    np.testing.assert_array_equal(result,np.zeros((1,6),np.float32))
    torch.manual_seed(330600);net=RetrievalNet([10,12,18]).cuda().eval()
    xs=[torch.randn(6,n,device='cuda') for n in [10,12,18]];perm=np.array([3,0,1,5,4,2])
    with torch.inference_mode():
        pred=net(*xs).cpu().numpy();one=net(*[x[:1] for x in xs]).cpu().numpy()
        pp=net(*[x[perm] for x in xs]).cpu().numpy()
    # TF32 GEMM picks different tiling for singleton vs batched execution.
    # Bound numerical differences; never compare ranks or metrics with this tolerance.
    np.testing.assert_allclose(pred[:1],one,rtol=5e-4,atol=1e-4)
    np.testing.assert_allclose(pred[perm],pp,rtol=5e-4,atol=1e-4)
    b.save(OUT/'isolation_audit.json',dict(signature=signature(),tasks=records,
        prohibited_singleton_neighbor_weight_zero=True,query_batch_invariance_within_tf32_tolerance=True,
        batch_absolute_tolerance=1e-4,batch_singleton_max_abs_difference=float(np.max(np.abs(pred[:1]-one))),test_labels_used=False))
    del net;torch.cuda.empty_cache()

def main():
    b.remote_only();torch.backends.cuda.matmul.allow_tf32=True;d=b.build_data()
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature() and report['status']=='EVALUATED'
    audit(d,other_cache(d,[0]));other=other_cache(d,[0,1])
    compact=json.loads((OUT/'test_predictions_compact.json').read_text());records=[]
    for r in report['tasks']:
        if r['selection'] is None:continue
        task=r['task'];k=d['tasks'].index(task);m=r['metric'];cls=m in ['AUROC','AUPRC']
        rows=np.flatnonzero((d['tid']==k)&(d['split']<2));query=np.flatnonzero((d['tid']==k)&(d['split']==2))
        x,_,_,tf,_=features(d,k,rows,query,[0,1],other);dest=OUT/task/'final';parts=[]
        qx=[torch.tensor(x['query'][n],device='cuda') for n in ['fp','dense','aux']]
        for seed in P['seeds']:
            snapshots=[]
            for epoch in P['snapshots']:
                ck=torch.load(dest/f'neural_{seed}_{epoch}.pt',map_location='cpu',weights_only=False)
                assert ck['signature']==signature()
                net=RetrievalNet(ck['dimensions']).cuda().eval();net.load_state_dict(ck['state'])
                with torch.inference_mode():p=np.concatenate([net(*[a[start:start+512] for a in qx]).cpu().numpy() for start in range(0,len(query),512)])
                snapshots.append(p if m=='Spearman' else ck['target'].decode(p));del net
            parts.append(np.mean(snapshots,0))
        pred=[np.mean(parts,0)];del qx;torch.cuda.empty_cache()
        klass=CatBoostClassifier if cls else CatBoostRegressor
        for family in range(2):
            values=[]
            for seed in P['seeds']:
                model=klass();model.load_model(str(dest/f'boost_{family}_{seed}.cbm'))
                p=model.predict_proba(x['query']['trees'][family])[:,1] if cls else model.predict(x['query']['trees'][family])
                values.append(p if cls or m=='Spearman' else tf.decode(p))
            pred.append(np.mean(values,0))
        np.testing.assert_allclose(np.stack(pred),compact[task],rtol=2e-5,atol=2e-5)
        records.append(dict(task=task,query_rows=len(query),all_model_reloads_passed=True))
        b.emit('v33_replay',task=task,passed=True)
    b.save(OUT/'model_replay.json',dict(status='FINAL_MODEL_REPLAY_PASSED',signature=signature(),tasks=records,local_gpu_training=False))
if __name__=='__main__':main()
