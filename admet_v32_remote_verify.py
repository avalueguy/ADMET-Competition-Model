"""Reload all final models for every selected task and replay test predictions."""
import json
import numpy as np
import torch
from catboost import CatBoostClassifier,CatBoostRegressor
import admet_v24_models as b
from admet_v32_run import ROOT,OUT,P,signature,features,ConditionalNet

def main():
    b.remote_only();d=b.build_data();report=json.loads((OUT/'analysis.json').read_text())
    assert report['signature']==signature() and report['status']=='EVALUATED' and len(report['tasks'])==22
    compact=json.loads((OUT/'test_predictions_compact.json').read_text());records=[]
    for record in report['tasks']:
        if record['selection'] is None:continue
        task=record['task'];k=d['tasks'].index(task);m=record['metric'];cls=m in ['AUROC','AUPRC']
        rows=np.flatnonzero((d['tid']==k)&(d['split']<2));query=np.flatnonzero((d['tid']==k)&(d['split']==2))
        x,_,_,_=features(d,k,rows,query,[0,1]);dest=OUT/task/'final';parts=[]
        qx=[torch.tensor(x['query'][key],device='cuda') for key in ['fp','dense','assay']]
        for seed in P['seeds']:
            snapshots=[]
            for epoch in P['snapshots']:
                ck=torch.load(dest/f'neural_{seed}_{epoch}.pt',map_location='cpu',weights_only=False)
                assert ck['signature']==signature()
                net=ConditionalNet(*ck['dimensions']).cuda().eval();net.load_state_dict(ck['state'])
                with torch.inference_mode():
                    p=np.concatenate([net(*[a[start:start+512] for a in qx]).cpu().numpy() for start in range(0,len(query),512)])
                snapshots.append(p if m=='Spearman' else ck['target'].decode(p));del net
            parts.append(np.mean(snapshots,0))
        pred=[np.mean(parts,0)];del qx;torch.cuda.empty_cache()
        tf=b.Target(d['y'][rows],m);klass=CatBoostClassifier if cls else CatBoostRegressor
        for family in range(2):
            values=[]
            for seed in P['seeds']:
                model=klass();model.load_model(str(dest/f'boost_{family}_{seed}.cbm'))
                p=model.predict_proba(x['query']['trees'][family])[:,1] if cls else model.predict(x['query']['trees'][family])
                values.append(p if cls or m=='Spearman' else tf.decode(p))
            pred.append(np.mean(values,0))
        np.testing.assert_allclose(np.stack(pred),compact[task],rtol=2e-5,atol=2e-5)
        records.append(dict(task=task,query_rows=len(query),all_three_family_reloads_passed=True))
        b.emit('v32_replay',task=task,passed=True)
    b.save(OUT/'model_replay.json',dict(status='FINAL_MODEL_REPLAY_PASSED',signature=signature(),tasks=records,local_gpu_training=False))
if __name__=='__main__':main()
