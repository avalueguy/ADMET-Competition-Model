"""Fit-only smooth molecular residual correction over the reported v84 anchor.

No external data or test labels. Historical proxies are not complete anchor
OOF and their prior label reuse prevents unbiased generalization estimates.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import admet_v77_ranking as b
import v83_residual as r83
import v84_neighbor_transfer as r84

OUT=ROOT/'admet_model_outputs_v85'
ANCHOR=ROOT/'admet_submissions/v84_neighbor_transfer/我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
ANCHOR_SHA='cf123205360536dbdcb88f97acb1c258570d74d9ddd24883f96d3332dfb80c71'
CONFIGS=[dict(name=f'{view}_a{alpha:g}',view=view,alpha=alpha)
    for view,alphas in [('chem',(.3,3.,30.)),('physical',(.3,3.,30.)),('assay',(.1,1.,10.))] for alpha in alphas]
STRENGTHS=(.25,.5,1.)
LANDMARKS=384
SEED=854026


def signature():
    return hashlib.sha256(''.join(b.sha(p) for p in (__file__,b.__file__,r83.__file__,r84.__file__)).encode()).hexdigest()


def emit(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True)


def load():
    d=b.load_data();d['legacy_folds']=np.load(ROOT/'recovered_history/legacy_v24_folds.npy')
    r83.TASK_NAMES=d['tasks'];assert b.sha(ANCHOR)==ANCHOR_SHA
    with np.load(ROOT/'admet_model_outputs_v67/features/official_chemistry.npz') as z:
        d['fingerprint']=z['x'][:,:4096].astype(np.float32)
    return d


def baseline(d,k,part):
    if part=='final':return np.array([float(r['y_pred']) for r in b.table(ANCHOR) if r['task']==d['tasks'][k]])
    p=r84.baseline(d,k,part)
    if d['tasks'][k]=='clearance_hepatocyte_az':
        cfg=next(c for c in r84.CONFIGS if c['name']=='combined_d3_t250')
        pred=r84.fit(d,k,part,cfg,r84.profiles(d,False))
        p=r84.mix(d['metrics'][k],p,pred,.5)
    return p


def base_state(d,k,part):
    fit,q=r83.split(d,k,part)
    base=r83.raw_proxy(d,fit)
    qp=(np.array([float(r['y_pred']) for r in b.table(r84.ORIGINAL) if r['task']==d['tasks'][k]])
        if part=='final' else r83.raw_proxy(d,q))
    y=d['y'][fit];metric=d['metrics'][k]
    if metric in ('AUROC','AUPRC'):
        z,qz=r83.ranklog(base),r83.ranklog(qp);beta=r83.calibrate(y,z)
        off,qoff=beta[0]+beta[1]*z,beta[0]+beta[1]*qz
        target=y;scale=1.;state=dict(beta=beta.tolist(),scale=scale)
    elif metric=='MAE':
        center=float(np.median(y));scale=max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-3)
        z,qz=(base-center)/scale,(qp-center)/scale
        target=(y-base)/scale;off,qoff=np.zeros(len(fit)),qp
        state=dict(center=center,scale=scale)
    else:
        z,qz=r83.ranknorm(base),r83.ranknorm(qp);xx=np.c_[np.ones(len(z)),z]
        beta=np.linalg.solve(xx.T@xx+np.diag([1e-8,1.]),xx.T@r83.ranknorm(y));beta[1]=max(beta[1],.01)
        off,qoff=xx@beta,beta[0]+beta[1]*qz
        target=r83.ranknorm(y)-off;scale=1.;state=dict(beta=beta.tolist(),scale=scale)
    return fit,q,base,qp,z,qz,off,qoff,target,state


def compute_design(d,k,part,view):
    fit,q,base,qp,z,qz,off,qoff,target,state=base_state(d,k,part)
    folder=OUT/'design'/d['tasks'][k]/part/view;path=folder/'design.npz'
    if path.exists():
        zz=np.load(path);assert str(zz['signature'])==signature()
        np.testing.assert_array_equal(zz['fit'],fit);np.testing.assert_array_equal(zz['query'],q)
        return dict(zz)
    folder.mkdir(parents=True,exist_ok=True)
    allrows=np.r_[fit,q];mi=d['mi'][allrows]
    extra={}
    if view=='assay':
        aux,cols,source,counts=r83.aux(d,k,part=='final');assert k not in cols
        aa=aux[fit];values=aa[:,:len(cols)]
        # Finite measurements only; missingness remains a separate covariate.
        center=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in values.T])
        std=np.array([np.std(col[np.isfinite(col)]) if np.isfinite(col).any() else 1. for col in values.T]).clip(.05)
        def trans(rows,score):
            a=aux[rows];mask=a[:,len(cols):];vals=np.nan_to_num((a[:,:len(cols)]-center)/std).clip(-5,5)*mask
            return np.c_[np.ones(len(rows)),vals,mask,score]
        x,qx=trans(fit,z),trans(q,qz)
        extra.update(aux_center=center,aux_std=std,aux_source=source,aux_cols=cols)
    else:
        # Pick unique canonical fitting molecules as fixed Nyström landmarks.
        unique,first=np.unique(d['canonical'][fit],return_index=True)
        rng=np.random.default_rng(SEED+k)
        chosen=rng.permutation(len(unique))[:min(LANDMARKS,len(unique))]
        landmarks=fit[first[chosen]];lm=d['mi'][landmarks]
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        if view=='chem':
            f=d['fingerprint'];query=torch.as_tensor(f[mi],device='cuda');reference=torch.as_tensor(f[lm],device='cuda')
            dot=query@reference.T;norms=query.square().sum(1);rn=reference.square().sum(1)
            kernel=(dot/(norms[:,None]+rn[None]-dot).clamp_min(1e-8)).clamp(0,1)
            kernel=(.6*kernel+.4*kernel.square()).cpu().numpy().astype(float)
            extra.update(landmarks=landmarks)
        else:
            full=d['views']['physchem'];xx=full[d['mi'][fit]].astype(float)
            med=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in xx.T])
            xx=np.where(np.isfinite(xx),xx,med)
            mean=xx.mean(0);std=xx.std(0).clip(.05)
            feat=((np.where(np.isfinite(full),full,med)-mean)/std).clip(-5,5).astype(np.float32)
            query=torch.as_tensor(feat[mi],device='cuda');reference=torch.as_tensor(feat[lm],device='cuda')
            dist=(query.square().sum(1)[:,None]+reference.square().sum(1)[None]-2*query@reference.T).clamp_min(0)/feat.shape[1]
            kernel=(.5*torch.exp(-dist/.5)+.5*torch.exp(-dist/2.)).cpu().numpy().astype(float)
            extra.update(landmarks=landmarks,physical_median=med,physical_mean=mean,physical_std=std)
        # Stable whitening makes the ridge penalty a molecular-kernel penalty.
        fit_lookup={int(row):i for i,row in enumerate(fit)}
        li=np.array([fit_lookup[int(row)] for row in landmarks]);w=kernel[li]
        eig,vec=np.linalg.eigh((w+w.T)*.5)
        keep=eig>1e-6;whitener=vec[:,keep]/np.sqrt(eig[keep])
        features=kernel@whitener
        # Add constant and low-dimensional base score for global calibration.
        features=np.c_[np.ones(len(features)),features,np.r_[z,qz]]
        x,qx=features[:len(fit)],features[len(fit):]
        extra.update(whitener=whitener)
        del query,reference
        torch.cuda.empty_cache()
    output=dict(signature=np.array(signature()),fit=fit,query=q,x=x,qx=qx,base_fit=base,base_query=qp,
        off=off,qoff=qoff,target=target,**extra)
    np.savez_compressed(path,**output)
    b.save(folder/'state.json',state)
    return output


def solve_model(x,target,off,metric,alpha):
    """Penalized logistic IRLS, robust MAE IRLS, or rank residual ridge."""
    x=np.asarray(x,float);n,p=x.shape
    # Mean-loss ridge for assay features; kernel ridge for Nyström features.
    penalty=float(alpha)*n if p<100 else float(alpha)
    reg=np.full(p,penalty);reg[0]*=.01
    coef=np.zeros(p)
    for _ in range(12 if metric in ('AUROC','AUPRC') else 8 if metric=='MAE' else 1):
        if metric in ('AUROC','AUPRC'):
            eta=off+x@coef;prob=expit(eta);weight=(prob*(1-prob)).clip(.03)
            working=x@coef+(target-prob)/weight
        elif metric=='MAE':
            residual=target-x@coef;weight=1/np.sqrt(residual**2+.05**2)
            weight=weight.clip(.1,20.);working=target
        else:weight=np.ones(n);working=target
        lhs=x.T@(weight[:,None]*x);lhs.flat[::p+1]+=reg
        new=np.linalg.solve(lhs,x.T@(weight*working))
        if np.max(np.abs(new-coef))<1e-7:coef=new;break
        coef=new
    assert np.isfinite(coef).all()
    return coef


def fit(d,k,part,cfg):
    folder=OUT/'models'/d['tasks'][k]/part/cfg['name'];path=folder/'predictions.npz'
    if path.exists():
        zz=np.load(path);assert str(zz['signature'])==signature();return dict(zz)
    started=time.monotonic();design=compute_design(d,k,part,cfg['view'])
    coef=solve_model(design['x'],design['target'],design['off'],d['metrics'][k],cfg['alpha'])
    delta=design['qx']@coef;folder.mkdir(parents=True,exist_ok=True)
    output=dict(signature=np.array(signature()),coef=coef,query=design['query'],delta=delta)
    np.savez_compressed(path,**output)
    b.save(folder/'metadata.json',dict(signature=signature(),config=cfg,task=d['tasks'][k],part=part,
        prediction_sha256=b.sha(path),seconds=time.monotonic()-started))
    return output


def predict(d,k,part,cfg,strength):
    model=fit(d,k,part,cfg);design=compute_design(d,k,part,cfg['view'])
    state=json.loads((OUT/'design'/d['tasks'][k]/part/cfg['view']/'state.json').read_text())
    metric=d['metrics'][k];raw=design['qoff']+strength*model['delta']*state['scale']
    pred=expit(raw) if metric in ('AUROC','AUPRC') else raw
    if d['tasks'][k]=='ppbr_az':pred=pred.clip(0,100)
    return pred


def develop(d):
    OUT.mkdir(exist_ok=True)
    b.save(OUT/'protocol.json',dict(version=85,signature=signature(),anchor_sha256=ANCHOR_SHA,
        baseline='v84',baseline_official_score_user_reported=50.880,configs=CONFIGS,strengths=STRENGTHS,
        select_parts=['fold0','fold1'],confirm_parts=['fold2','valid'],landmarks=LANDMARKS,seed=SEED,
        selection='Max minimum gain on fold0/1. Minimum practical margin .003 relative MAE, .0005 AUROC/AUPRC, .001 Spearman.',
        confirmation='Freeze task/config/strength, require all four parts positive; no fallback tuning on failed confirmation.',
        test_labels_used=False,external_data=False,new_pretrained_weights=False,
        limitations='Historical base proxy, not full exact v84 OOF; target labels in historical proxies reused. Conditional assay branch excludes own task, includes official other-task train values.'))
    records=[]
    for k,task in enumerate(d['tasks']):
        metric=d['metrics'][k];base={part:baseline(d,k,part) for part in ('fold0','fold1')}
        for cfg in CONFIGS:
            for strength in STRENGTHS:
                parts=[]
                for part in ('fold0','fold1'):
                    pred=predict(d,k,part,cfg,strength);q=r83.split(d,k,part)[1]
                    parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],base[part]),
                        candidate=r83.score(metric,d['y'][q],pred),gain=r83.gain(metric,d['y'][q],base[part],pred)))
                records.append(dict(task=task,metric=metric,config=cfg,strength=strength,parts=parts,
                    objective=min(p['gain'] for p in parts)))
        chosen=max((r for r in records if r['task']==task),key=lambda r:r['objective'])
        emit(stage='development',task=task,best=chosen)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',records=records,signature=signature()))
    selected=[]
    for task in d['tasks']:
        chosen=max((r for r in records if r['task']==task),key=lambda r:r['objective'])
        margin=.003 if chosen['metric']=='MAE' else .001 if chosen['metric']=='Spearman' else .0005
        if chosen['objective']>margin:selected.append(chosen)
    b.save(OUT/'development.json',dict(status='COMPLETE',records=records,signature=signature()))
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=signature(),
        selected=selected,confirmation_inspected=False))
    emit(stage='frozen',selected=selected)


def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==signature()
    records=[]
    for chosen in frozen['selected']:
        k=d['tasks'].index(chosen['task']);metric=chosen['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];base=baseline(d,k,part);pred=predict(d,k,part,chosen['config'],chosen['strength'])
            parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],base),
                candidate=r83.score(metric,d['y'][q],pred),gain=r83.gain(metric,d['y'][q],base,pred)))
        record=dict(**chosen,confirmation=parts,accepted=all(p['gain']>0 for p in parts))
        records.append(record);emit(stage='confirmation',**record)
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=signature(),records=records,frozen_sha256=b.sha(OUT/'frozen_selection.json')))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=signature(),records=records,
        accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','confirm']);args=ap.parse_args();d=load()
    (develop if args.stage=='develop' else confirm)(d)
