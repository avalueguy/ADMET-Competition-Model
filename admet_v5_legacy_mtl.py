"""Rebuild historical MTL under a target fold's global scaffold exclusion."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from admet_features_v2 import features, SCHEMA, DescriptorPreprocessor
from admet_multitask import TaskGatedExperts

_DATA=None


def existing_features(smiles):
    smi=list(map(str,smiles))
    key=hashlib.sha256(json.dumps([SCHEMA,smi],sort_keys=True).encode()).hexdigest()
    for c in ['admet_model_outputs_v5/features','admet_model_outputs_v5_tail/features',
              'admet_model_outputs_v5_middle/features']:
        if (Path(c)/(key+'.npz')).exists():return features(smi,Path(c))
    return features(smi,Path('admet_v5_confirmation/features'))


def collect(tasks):
    global _DATA
    if _DATA is not None:return _DATA
    names=sorted(tasks)
    smi=sorted(set(str(s) for task in names for frame in tasks[task].values() for s in frame.Drug))
    lookup={s:i for i,s in enumerate(smi)}
    x=np.zeros((len(smi),4096+len(SCHEMA['descriptors'])),np.float32)
    canon=np.empty(len(smi),object);scaffolds=np.empty(len(smi),object)
    indices={}
    for task in names:
        frames=tasks[task]
        all_smi=pd.concat([df.Drug for df in frames.values()]).astype(str).tolist()
        xx,cc,ss=existing_features(all_smi)
        ix=np.array([lookup[s] for s in all_smi])
        x[ix]=xx;canon[ix]=cc;scaffolds[ix]=ss
        indices[task]={s:np.array([lookup[str(v)] for v in df.Drug]) for s,df in frames.items()}
    _DATA=(names,x,canon,scaffolds,indices)
    return _DATA


def fit_predict(task, fold, forbidden_scaffolds, smiles_hold, tasks, cache):
    names,x,canon,scaffolds,indices=collect(tasks)
    cache=Path(cache)/task/f'fold_{fold}'
    cache.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(sorted(forbidden_scaffolds)).encode()).hexdigest()
    if (cache/'predictions.npz').exists():
        with np.load(cache/'predictions.npz') as z:
            assert str(z['signature'])==signature
            return z['predictions']
    assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.manual_seed(1729)
    rng=np.random.default_rng(1729)
    forbidden=set(canon[np.concatenate([indices[n][s] for n in names for s in ['valid','test']])])
    official=pd.read_csv('admet_official/submission_template.csv').groupby('task').metric.first().to_dict()
    by_task=[];transforms=[];audits=[]
    for n in names:
        ix=indices[n]['train'];y=tasks[n]['train'].Y.to_numpy(float)
        mask=np.array([c not in forbidden and s not in forbidden_scaffolds for c,s in zip(canon[ix],scaffolds[ix])])
        ix,y=ix[mask],y[mask]
        assert not set(scaffolds[ix])&forbidden_scaffolds
        assert not set(canon[ix])&forbidden
        if not len(ix):raise ValueError(f'{n} no MTL records after scaffold exclusion')
        agg=pd.DataFrame(dict(canonical=canon[ix],ix=ix,y=y)).groupby('canonical').agg(ix=('ix','first'),y=('y','mean'))
        ix,y=agg.ix.to_numpy(int),agg.y.to_numpy(float)
        classification=official[n] in ['AUROC','AUPRC']
        log=not classification and official[n]=='Spearman' and y.min()>=0
        yy=np.log1p(y) if log else y.copy()
        mean,scale=(0.,1.) if classification else (float(yy.mean()),max(float(yy.std()),1e-6))
        by_task.append((ix,((yy-mean)/scale).astype(np.float32)))
        transforms.append(dict(classification=classification,log=log,mean=mean,scale=scale))
        audits.append(dict(source=n,records=len(ix),excluded_records=int((~mask).sum()),heldout_scaffold_overlap=0))
    training_ix=np.unique(np.concatenate([r[0] for r in by_task]))
    pre=DescriptorPreprocessor().fit(x[training_ix])
    xx=torch.as_tensor(pre.transform(x),device='cuda')
    net=TaskGatedExperts(x.shape[1],len(names)).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=32)
    class_tasks=torch.tensor([t['classification'] for t in transforms],device='cuda')
    snapshots=[]
    for ep in range(32):
        net.train()
        for _ in range(55):
            bix=[];by=[];bt=[]
            for t,(ix,yy) in enumerate(by_task):
                draw=rng.integers(len(ix),size=24)
                bix.extend(ix[draw]);by.extend(yy[draw]);bt.extend([t]*len(draw))
            yy=torch.tensor(np.array(by),dtype=torch.float32,device='cuda')
            tt=torch.tensor(bt,dtype=torch.long,device='cuda')
            p=net(xx[np.array(bix)],tt)
            loss=torch.where(class_tasks[tt],nn.functional.binary_cross_entropy_with_logits(p,yy,reduction='none'),
                             nn.functional.smooth_l1_loss(p,yy,reduction='none')).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3.)
            opt.step()
        scheduler.step()
        if ep+1 in [28,32]:snapshots.append({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
    xh,_,_=features(smiles_hold,Path('admet_v5_confirmation/features'))
    xh=torch.as_tensor(pre.transform(xh),device='cuda')
    t=names.index(task);tr=transforms[t];ps=[]
    with torch.inference_mode():
        for state in snapshots:
            net.load_state_dict(state);net.eval();p=[]
            for start in range(0,len(xh),512):
                chunk=xh[start:start+512]
                p.extend(net(chunk,torch.full((len(chunk),),t,device='cuda',dtype=torch.long)).cpu().numpy())
            p=np.array(p)
            if tr['classification']:p=1/(1+np.exp(-np.clip(p,-30,30)))
            else:
                p=p*tr['scale']+tr['mean']
                if tr['log']:p=np.expm1(np.clip(p,-30,30))
            ps.append(p)
    predictions=np.mean(ps,axis=0)
    torch.save(dict(states=snapshots,names=names,transforms=transforms,input_dim=x.shape[1],
                    signature=signature,epochs=32,seed=1729,device='cuda'),cache/'model.pt')
    joblib.dump(pre,cache/'preprocessor.joblib')
    pd.DataFrame(audits).to_csv(cache/'source_exclusion.csv',index=False)
    np.savez_compressed(cache/'predictions.npz',predictions=predictions,signature=signature)
    print(json.dumps(dict(task=task,fold=fold,stage='legacy_mtl_rebuilt',scaffold_exclusion=True)),flush=True)
    del xx,xh,net
    torch.cuda.empty_cache()
    return predictions
