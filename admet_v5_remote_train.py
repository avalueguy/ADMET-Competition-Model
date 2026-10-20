"""Remote experiment runner: scaffold CV, frozen ChemBERTa and neural heads.

This produces research candidates, never overwrites a release or publishes a
submission. The existing official valid split is reused and is labelled as
such. No fitted model sees test targets or auxiliary task labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import expit
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC, SVR
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from admet_features_v2 import features, DescriptorPreprocessor
from admet_upgrade_v3 import metrics, dominates, utility, get_metric


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Preprocessor:
    def fit(self, raw, emb):
        self.desc = DescriptorPreprocessor().fit(raw)
        self.mean = emb.mean(0)
        self.scale = np.maximum(emb.std(0), .05)
        return self

    def transform(self, raw, emb):
        x = self.desc.transform(raw)
        e = np.clip((emb-self.mean)/self.scale, -10, 10)
        return np.hstack([x, e]).astype(np.float32)


class NeuralHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, 256), nn.LayerNorm(256),
                                    nn.SiLU(), nn.Dropout(.25), nn.Linear(256,128),
                                    nn.SiLU(), nn.Dropout(.15), nn.Linear(128,1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def kernel_matrix(a, b):
    a, b = a[:,:4096].astype(np.float32), b[:,:4096].astype(np.float32)
    dots = a@b.T
    return (dots/np.maximum(a.sum(1)[:,None]+b.sum(1)[None,:]-dots,1)).astype(np.float64)


def fit_family(name, raw, emb, y, kind, metric, protocol):
    pre = Preprocessor().fit(raw, emb)
    x = pre.transform(raw, emb)
    log_target = kind == 'regression' and metric == 'Spearman' and np.min(y)>=0
    yy = np.log1p(y) if log_target else y.copy()
    mean = float(yy.mean()) if kind=='regression' else 0.
    scale = max(float(yy.std()), 1e-6) if kind=='regression' else 1.
    yy = (yy-mean)/scale
    bundle = dict(family=name, pre=pre, kind=kind, metric=metric,
                  log_target=log_target, mean=mean, scale=scale, models=[],
                  seeds=protocol['seeds'], encoder='admet_pretrained/chemberta_mlm')
    if name=='kernel':
        kernel=kernel_matrix(raw,raw)
        # decision values support ranking metrics; sigmoid used for probability range.
        model = (SVC(C=1.,kernel='precomputed',class_weight='balanced') if kind=='classification'
                 else SVR(C=3.,epsilon=.1,kernel='precomputed'))
        model.fit(kernel, yy)
        bundle.update(models=[model], reference_fp=raw[:,:4096].copy())
        return bundle
    for seed in protocol['seeds']:
        if name=='catboost':
            common=dict(iterations=650,depth=6,learning_rate=.04,l2_leaf_reg=6,
                        random_seed=seed,thread_count=6,verbose=False,allow_writing_files=False)
            model=(CatBoostClassifier(loss_function='Logloss',**common) if kind=='classification'
                   else CatBoostRegressor(loss_function='MAE',**common))
            # Dense molecular descriptors and frozen language embeddings complement fingerprints.
            model.fit(x[:,4096:], yy)
        else:
            torch.manual_seed(seed)
            rng=np.random.default_rng(seed)
            model=NeuralHead(x.shape[1]).cuda()
            xx=torch.as_tensor(x,device='cuda')
            target=torch.as_tensor(yy,dtype=torch.float32,device='cuda')
            opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.02)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,protocol['neural_epochs'])
            snapshots=[]
            for epoch in range(protocol['neural_epochs']):
                model.train()
                for start in range(0,len(x),128):
                    if start==0: order=rng.permutation(len(x))
                    ix=order[start:start+128]
                    p=model(xx[ix])
                    loss=(nn.functional.binary_cross_entropy_with_logits(p,target[ix])
                          if kind=='classification' else nn.functional.smooth_l1_loss(p,target[ix]))
                    opt.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),3.)
                    opt.step()
                sched.step()
                if epoch+1 in {protocol['neural_epochs']-5,protocol['neural_epochs']}:
                    snapshots.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            bundle['models'].extend(snapshots)
            bundle['input_dim']=x.shape[1]
            del model,xx,target
            torch.cuda.empty_cache()
            continue
        bundle['models'].append(model)
    return bundle


def predict(bundle, raw, emb):
    x=bundle['pre'].transform(raw,emb); preds=[]
    for model in bundle['models']:
        if bundle['family']=='kernel':
            kk=kernel_matrix(raw,bundle['reference_fp'])
            p=expit(model.decision_function(kk)) if bundle['kind']=='classification' else model.predict(kk)
        elif bundle['family']=='catboost':
            p=model.predict_proba(x[:,4096:])[:,1] if bundle['kind']=='classification' else model.predict(x[:,4096:])
        else:
            net=NeuralHead(bundle['input_dim']).cuda(); net.load_state_dict(model); net.eval()
            ps=[]
            with torch.inference_mode():
                for start in range(0,len(x),256):
                    p=net(torch.as_tensor(x[start:start+256],device='cuda'))
                    ps.append(p.cpu().numpy())
            p=np.concatenate(ps)
            if bundle['kind']=='classification': p=expit(p)
            del net
        if bundle['kind']=='regression':
            p=p*bundle['scale']+bundle['mean']
            if bundle['log_target']: p=np.expm1(np.clip(p,-30,30))
        preds.append(p)
    return np.mean(preds,axis=0)


def run_task(task, protocol, out):
    root=Path('admet_competition_data/data_benchmark')/task
    frames={s:pd.read_csv(root/(s+'.csv')) for s in ['train','valid','test']}
    n,nv=len(frames['train']),len(frames['valid'])
    smi=pd.concat([df.Drug for df in frames.values()]).astype(str).tolist()
    raw,canon,groups=features(smi,out/'features')
    with np.load(Path('admet_pretrained/embeddings')/(task+'.npz')) as z:
        if str(z['digest'])!=hashlib.sha256(json.dumps(smi).encode()).hexdigest():
            raise ValueError('Encoder row checksum mismatch')
        emb=z['x']
    assert len(emb)==len(raw)
    y=frames['train'].Y.to_numpy(float); yv=frames['valid'].Y.to_numpy(float)
    kind='classification' if set(np.unique(y)).issubset({0,1}) else 'regression'
    metric=get_metric(task); dest=out/task; dest.mkdir(parents=True,exist_ok=True)
    folds=list(GroupKFold(n_splits=protocol['cv_folds'],shuffle=True,
                         random_state=protocol['cv_seed']).split(raw[:n],y,groups[:n]))
    partition=np.full(n,-1)
    for fold,(_,iv) in enumerate(folds): partition[iv]=fold
    pd.DataFrame(dict(row=np.arange(n),fold=partition,canonical=canon[:n],
                      scaffold=groups[:n])).to_csv(dest/'folds.csv',index=False)
    names=[name for name in protocol['candidates'] if name!='kernel' or n<=protocol['kernel_max_train']]
    summary=[]
    for name in names:
        rows=[]; oof=np.full(n,np.nan)
        for fold,(it,iv) in enumerate(folds):
            assert not set(np.asarray(groups)[it])&set(np.asarray(groups)[iv])
            assert not set(np.asarray(canon)[it])&set(np.asarray(canon)[iv])
            if kind=='classification' and (len(np.unique(y[iv]))<2 or len(np.unique(y[it]))<2):
                raise ValueError('Undefined classification fold; do not silently replace split')
            model=fit_family(name,raw[it],emb[it],y[it],kind,metric,protocol)
            p=predict(model,raw[iv],emb[iv]);oof[iv]=p
            m=metrics(kind,y[iv],p)
            rows.append(dict(fold=fold,**m))
            print(json.dumps(dict(task=task,family=name,fold=fold,**m)),flush=True)
        assert np.isfinite(oof).all()
        pd.DataFrame(rows).to_csv(dest/(name+'_cv.csv'),index=False)
        pd.DataFrame(dict(row=np.arange(n),Y_true=y,Y_pred=oof,fold=partition)).to_csv(dest/(name+'_oof.csv'),index=False)
        summary.append(dict(family=name,mean_utility=float(np.mean([utility(metric,r) for r in rows])),
                            sd_utility=float(np.std([utility(metric,r) for r in rows]))))
    chosen=max(summary,key=lambda r:r['mean_utility'])['family']
    # Lock before accessing validation labels for evaluation. No alpha search.
    config=dict(task=task,kind=kind,metric=metric,chosen=chosen,cv_summary=summary,
                protocol_sha256=digest('admet_v5_protocol.json'),
                decision_source='train scaffold CV only',alpha=protocol['validation_blend_weight'])
    (dest/'locked_selection.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    fitted=fit_family(chosen,raw[:n],emb[:n],y,kind,metric,protocol)
    candidate=predict(fitted,raw[n:n+nv],emb[n:n+nv])
    old=pd.read_csv(Path('admet_release_v4')/task/'validation_predictions.csv')
    assert np.allclose(old.Y_true,yv)
    alpha=config['alpha']; mixed=(1-alpha)*old.Y_pred.to_numpy(float)+alpha*candidate
    before=metrics(kind,yv,old.Y_pred); after=metrics(kind,yv,mixed)
    provisional=dominates(kind,before,after) and utility(metric,after)>utility(metric,before)+1e-8
    result=dict(**config,valid_before=before,valid_candidate=metrics(kind,yv,candidate),
                valid_blend=after,provisional_valid_pass=provisional,
                official_score=None,publication_authorized=False,
                limitation='Reused official valid; full historical ensemble not retrained in CV; further matched baseline confirmation required')
    pd.DataFrame(dict(Drug_ID=frames['valid'].Drug_ID,Y_true=yv,baseline=old.Y_pred,
                      candidate=candidate,blend=mixed)).to_csv(dest/'validation_comparison.csv',index=False)
    joblib.dump(fitted,dest/'validation_model.joblib',compress=3)
    (dest/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result),flush=True)
    # No blind prediction or release is created until matched baseline confirmation.
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tasks',nargs='*'); ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if os.name=='nt' or not Path('/root/autodl-tmp/admet_20260925').is_dir():
        raise RuntimeError('This experiment is restricted to the requested remote workspace')
    if not torch.cuda.is_available(): raise RuntimeError('Remote CUDA required; no automatic local/CPU fallback')
    torch.set_num_threads(6)
    protocol=json.loads(Path('admet_v5_protocol.json').read_text())
    out=Path('admet_model_outputs_v5');out.mkdir(exist_ok=True)
    names=args.tasks or sorted(p.name for p in Path('admet_competition_data/data_benchmark').iterdir())
    with threadpool_limits(limits=6):
        for task in names:
            if args.resume and (out/task/'result.json').exists(): continue
            run_task(task,protocol,out)


if __name__=='__main__': main()
