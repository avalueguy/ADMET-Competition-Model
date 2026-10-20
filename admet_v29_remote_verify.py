"""Reload every final source model actually selected for the combined release."""
import json
from pathlib import Path
import numpy as np
import torch
import admet_v24_models as b
ROOT=Path(__file__).resolve().parent

def graph_replay(d):
    from admet_v27_models import graphs,pack,BondNet,P
    g=graphs(d);query=np.flatnonzero(d['split']==2);molecules=np.unique(d['mi'][query]);parts=[]
    for seed in P['final_seeds']:
        seedparts=[]
        for step in P['snapshot_steps']:
            ck=torch.load(ROOT/f'admet_model_outputs_v27/final/{seed}/checkpoint_{step}.pt',map_location='cpu',weights_only=False)
            net=BondNet(*ck['dimensions']).cuda().eval();net.load_state_dict(ck['state'])
            raw=np.concatenate([d['desc'],d['emb']],1);features=np.concatenate([d['fp2'],d['fcfp'],ck['preprocessor'].transform(raw)],1)
            features=torch.tensor(features,device='cuda');values=np.empty((2,len(d['smiles']),len(d['tasks'])),np.float32)
            with torch.inference_mode():
                for start in range(0,len(molecules),96):
                    mm=molecules[start:start+96];values[:,mm]=net(pack(g,mm),features[mm]).cpu().numpy()
            out=np.empty((2,len(query)),float)
            for k,t in enumerate(d['tasks']):
                ix=np.flatnonzero(d['tid'][query]==k);v=values[:,d['mi'][query[ix]],k]
                out[:,ix]=v if d['metrics'][k]=='Spearman' else ck['target'][k].decode(v)
            seedparts.append(out);del net,features;torch.cuda.empty_cache()
        parts.append(np.mean(seedparts,0))
    pred=np.mean(parts,0)
    expected=json.loads((ROOT/'admet_model_outputs_v27/test_predictions_compact.json').read_text())
    for k,t in enumerate(d['tasks']):np.testing.assert_allclose(pred[:,d['tid'][query]==k],expected[t],rtol=3e-5,atol=3e-5)
    return True

def mae_replay(d,task,variant):
    from admet_v28_run import P,Net
    from catboost import CatBoostRegressor
    k=d['tasks'].index(task);rows=np.flatnonzero((d['tid']==k)&(d['split']<2));query=np.flatnonzero((d['tid']==k)&(d['split']==2))
    dest=ROOT/'admet_model_outputs_v28'/task/'final';parts=[]
    dense=np.concatenate([d['desc'],d['emb']],1)
    if variant==0:
        for seed in P['seeds']:
            predictions=[]
            for epoch in P['snapshots']:
                ck=torch.load(dest/f'mlp_{seed}_{epoch}.pt',map_location='cpu',weights_only=False)
                raw=np.concatenate([d['fp2'],d['fcfp'],ck['preprocessor'].transform(dense)],1)
                net=Net(raw.shape[1]).cuda().eval();net.load_state_dict(ck['state'])
                x=torch.tensor(raw[d['mi'][query]],device='cuda')
                with torch.inference_mode():p=np.concatenate([net(x[start:start+512]).cpu().numpy() for start in range(0,len(query),512)])
                predictions.append(ck['target'].decode(p));del net,x
            parts.append(np.mean(predictions,0))
    else:
        pre=b.Preprocessor().fit(dense[np.unique(d['mi'][rows])]);x=pre.transform(dense)[d['mi'][query]];tf=b.Target(d['y'][rows],'MAE')
        for seed in P['seeds']:
            model=CatBoostRegressor();model.load_model(str(dest/f'boost_{seed}.cbm'));parts.append(tf.decode(model.predict(x)))
    expected=json.loads((ROOT/'admet_model_outputs_v28/test_predictions_compact.json').read_text())[task][variant]
    np.testing.assert_allclose(np.mean(parts,0),expected,rtol=3e-5,atol=3e-5);torch.cuda.empty_cache();return True

def assay_replay(d,task,variant):
    from admet_v31_run import auxiliary,P
    from catboost import CatBoostRegressor,CatBoostClassifier
    k=d['tasks'].index(task);rows=np.flatnonzero((d['tid']==k)&(d['split']<2));query=np.flatnonzero((d['tid']==k)&(d['split']==2))
    aux,count,_=auxiliary(d,[0,1],k);raw=np.concatenate([d['desc'],d['emb']],1)
    pre=b.Preprocessor().fit(raw[np.unique(d['mi'][rows])]);x=pre.transform(raw)
    features=np.concatenate([x[d['mi']],aux],1) if variant==0 else np.concatenate([x[d['mi'],:d['desc'].shape[1]],aux],1)
    m=d['metrics'][k];cls=m in ['AUROC','AUPRC'];klass=CatBoostClassifier if cls else CatBoostRegressor;tf=b.Target(d['y'][rows],m);parts=[]
    for seed in P['seeds']:
        model=klass();model.load_model(str(ROOT/'admet_model_outputs_v31'/task/'final'/f'family{variant}_{seed}.cbm'))
        q=model.predict_proba(features[query])[:,1] if cls else model.predict(features[query]);parts.append(q if cls or m=='Spearman' else tf.decode(q))
    expected=json.loads((ROOT/'admet_model_outputs_v31/test_predictions_compact.json').read_text())[task][variant]
    np.testing.assert_allclose(np.mean(parts,0),expected,rtol=3e-5,atol=3e-5);return True

if __name__=='__main__':
    b.remote_only();d=b.build_data();report=json.loads((ROOT/'admet_model_outputs_v29/analysis.json').read_text())
    sources=set((r['task'],s['version'],s['variant']) for r in report['tasks'] if r['selection'] for s in r['selection']['sources'])
    records=[]
    if any(version==27 for task,version,variant in sources):graph_replay(d);records.append(dict(version=27,model_reload_replay=True))
    for task,version,variant in sorted(sources):
        if version==27:continue
        (mae_replay if version==28 else assay_replay)(d,task,variant)
        records.append(dict(task=task,version=version,variant=variant,model_reload_replay=True))
    b.save(ROOT/'admet_model_outputs_v29/model_replay.json',dict(status='FINAL_SELECTED_MODEL_REPLAY_PASSED',sources=records,local_training=False))
    print(json.dumps(records))
