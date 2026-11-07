"""Official-only contextual assay kernels over the user-reported v86 anchor.

Fit-only Nyström features model smooth assay interactions and their product
with molecular similarity. Historical endpoint proxies remain approximate.
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
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import admet_v77_ranking as b
import v83_residual as r83
import v84_neighbor_transfer as r84
import v85_smooth_residual as r85
import v86_metric_ranking as r86
import v87_metric_ranking as r87

OUT=ROOT/'admet_model_outputs_v88'
ANCHOR=r87.ANCHOR
ANCHOR_SHA=r87.ANCHOR_SHA
LANDMARKS=384
SEED=884026
CONFIGS=[dict(name=f'{view}_a{alpha:g}',view=view,alpha=alpha)
    for view in ('measured','phenotype','product') for alpha in (.3,3.,30.)]
WEIGHTS=(.25,.5,1.)


def signature():
    return hashlib.sha256(''.join(b.sha(p) for p in (__file__,b.__file__,r83.__file__,r84.__file__,r85.__file__,r86.__file__,r87.__file__)).encode()).hexdigest()


def emit(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True)


def load():
    d=r85.load();assert b.sha(ANCHOR)==ANCHOR_SHA
    return d


def baseline(d,k,part):return r87.baseline(d,k,part)


def source_design(d,k,part):
    """Target assay values and all its neighbor profile fields are absent."""
    fit,q,base,qp,z,qz,off,qoff,target,state=r85.base_state(d,k,part)
    aux,cols,source,counts=r83.aux(d,k,part=='final');assert k not in cols
    # Median/IQR and imputation are fitted separately in every outer fit.
    vals=aux[fit,:len(cols)]
    center=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in vals.T])
    scale=np.array([np.std(c[np.isfinite(c)]) if np.isfinite(c).any() else 1. for c in vals.T]).clip(.05)
    def measured(rows):
        a=aux[rows];mask=a[:,len(cols):]
        value=np.nan_to_num((a[:,:len(cols)]-center)/scale).clip(-5,5)*mask
        return np.c_[value,mask]
    rows=np.r_[fit,q];meas=measured(rows)
    profile=r84.profiles(d,part=='final')
    # Full 5-field target neighbor slice is excluded, even though cached.
    neighbor=profile[d['mi'][rows]][:,cols,:].reshape(len(rows),-1).astype(float)
    train=neighbor[:len(fit)];ncenter=np.median(train,axis=0);nscale=train.std(0).clip(.05)
    neighbor=((neighbor-ncenter)/nscale).clip(-5,5)
    physical=d['views']['physchem'][d['mi'][rows]].astype(float);train=physical[:len(fit)]
    pmedian=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in train.T])
    filled=np.where(np.isfinite(physical),physical,pmedian);pmean=filled[:len(fit)].mean(0);pscale=filled[:len(fit)].std(0).clip(.05)
    physical=((filled-pmean)/pscale).clip(-5,5)
    return dict(fit=fit,query=q,rows=rows,source=source,cols=np.array(cols),known_counts=counts[q],
        measured=meas.astype(np.float32),neighbor=neighbor.astype(np.float32),physical=physical.astype(np.float32),
        measured_center=center,measured_scale=scale,neighbor_center=ncenter,neighbor_scale=nscale,
        physical_median=pmedian,physical_mean=pmean,physical_scale=pscale,
        base_fit=base,base_query=qp,z=z,qz=qz,off=off,qoff=qoff,target=target,state=state)


def kernels(d,a,k):
    """Three PSD contextual kernels, geometry and landmarks use fit rows only."""
    unique,first=np.unique(d['canonical'][a['fit']],return_index=True)
    chosen=np.random.default_rng(SEED+k).permutation(len(unique))[:min(LANDMARKS,len(unique))]
    li=first[chosen];landmarks=a['fit'][li]
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    def rbf(full):
        x=torch.as_tensor(full,device='cuda');ref=x[li]
        dist=(x.square().sum(1)[:,None]+ref.square().sum(1)[None]-2*x@ref.T).clamp_min(0)/x.shape[1]
        return .5*torch.exp(-dist/.5)+.5*torch.exp(-dist/2.)
    mk=rbf(a['measured'])
    # Neighbor profiles alone can generalize to molecules without same-molecule
    # measurements; the measured kernel retains availability information.
    pheno=.5*mk+.5*rbf(a['neighbor'])
    fp=torch.as_tensor(d['fingerprint'][d['mi'][a['rows']]],device='cuda');ref=fp[li]
    dot=fp@ref.T;tan=(dot/(fp.square().sum(1)[:,None]+ref.square().sum(1)[None]-dot).clamp_min(1e-8)).clamp(0,1)
    chemistry=.8*(.6*tan+.4*tan.square())+.2*rbf(a['physical'])
    output=dict(measured=mk,phenotype=pheno,product=.25*pheno+.75*pheno*chemistry)
    return {key:value.cpu().numpy().astype(float) for key,value in output.items()},li,landmarks


def compute_design(d,k,part,view):
    folder=OUT/'design'/d['tasks'][k]/part;path=folder/view/'design.npz'
    if path.exists():
        zz=np.load(path);assert str(zz['signature'])==signature()
        np.testing.assert_array_equal(zz['query'],r83.split(d,k,part)[1])
        return dict(zz)
    started=time.monotonic();a=source_design(d,k,part);ks,li,landmarks=kernels(d,a,k)
    for name,kernel in ks.items():
        w=kernel[li];eig,vec=np.linalg.eigh((w+w.T)*.5)
        keep=eig>1e-5;whitener=vec[:,keep]/np.sqrt(eig[keep]);features=kernel@whitener
        features=np.c_[np.ones(len(a['rows'])),features,np.r_[a['z'],a['qz']]]
        output=dict(signature=np.array(signature()),fit=a['fit'],query=a['query'],source=a['source'],cols=a['cols'],
            known_counts=a['known_counts'],landmarks=landmarks,landmark_positions=li,whitener=whitener,
            x=features[:len(a['fit'])],qx=features[len(a['fit']):],
            **{key:a[key] for key in ('measured_center','measured_scale','neighbor_center','neighbor_scale',
                'physical_median','physical_mean','physical_scale','base_fit','base_query','off','qoff','target')})
        dest=folder/name;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'design.npz',**output)
        b.save(dest/'state.json',a['state'])
    torch.cuda.empty_cache();emit(stage='design',task=d['tasks'][k],part=part,seconds=time.monotonic()-started)
    return dict(np.load(path))


def fit(d,k,part,cfg):
    folder=OUT/'models'/d['tasks'][k]/part/cfg['name'];path=folder/'predictions.npz'
    if path.exists():
        zz=np.load(path);assert str(zz['signature'])==signature();return dict(zz)
    a=compute_design(d,k,part,cfg['view'])
    # Explicit kernel regularization regardless of effective Nyström rank.
    # Do not reuse v85's dimension-dependent assay penalty heuristic.
    x=np.asarray(a['x'],float);target=a['target'];off=a['off'];metric=d['metrics'][k]
    p=x.shape[1];reg=np.full(p,float(cfg['alpha']));reg[0]*=.01
    coef=np.zeros(p)
    for _ in range(15 if metric in ('AUROC','AUPRC') else 12 if metric=='MAE' else 1):
        if metric in ('AUROC','AUPRC'):
            prob=expit(off+x@coef);weight=(prob*(1-prob)).clip(.03);working=x@coef+(target-prob)/weight
        elif metric=='MAE':
            weight=(1/np.sqrt((target-x@coef)**2+.05**2)).clip(.1,20);working=target
        else:weight=np.ones(len(x));working=target
        lhs=x.T@(weight[:,None]*x);lhs.flat[::p+1]+=reg
        new=np.linalg.solve(lhs,x.T@(weight*working))
        if np.max(np.abs(new-coef))<1e-7:coef=new;break
        coef=new
    delta=a['qx']@coef;assert np.isfinite(delta).all();folder.mkdir(parents=True,exist_ok=True)
    output=dict(signature=np.array(signature()),coef=coef,query=a['query'],delta=delta)
    np.savez_compressed(path,**output)
    b.save(folder/'metadata.json',dict(signature=signature(),task=d['tasks'][k],part=part,config=cfg,
        predictions_sha256=b.sha(path),design_sha256=b.sha(OUT/'design'/d['tasks'][k]/part/cfg['view']/'design.npz')))
    return output


def expert(d,k,part,cfg):
    m=fit(d,k,part,cfg);a=compute_design(d,k,part,cfg['view'])
    state=json.loads((OUT/'design'/d['tasks'][k]/part/cfg['view']/'state.json').read_text())
    raw=a['qoff']+m['delta']*state['scale']
    p=expit(raw) if d['metrics'][k] in ('AUROC','AUPRC') else raw
    return p.clip(0,100) if d['tasks'][k]=='ppbr_az' else p


def mix(metric,base,new,w):
    return (1-w)*base+w*new if metric=='MAE' else (1-w)*rankdata(base)/len(base)+w*rankdata(new)/len(new)


def develop(d):
    OUT.mkdir(exist_ok=True)
    b.save(OUT/'protocol.json',dict(version=88,signature=signature(),baseline='v86',anchor_sha256=ANCHOR_SHA,
        baseline_official_score_user_reported=50.900,configs=CONFIGS,weights=WEIGHTS,landmarks=LANDMARKS,seed=SEED,
        selection_parts=['fold0','fold1'],confirmation_parts=['fold2','valid'],
        selection='Max minimum gain. Require >.003 relative MAE, >.001 Spearman, >.0005 AUROC/AUPRC on both folds.',
        confirmation='Frozen config and blend; require all four gains positive. No confirmation fallback tuning.',
        new_direction='Smooth other-assay interactions and their product with molecular chemistry',
        target_assay_measurements_excluded=True,target_neighbor_profile_excluded=True,
        data_sources='Official train auxiliary measurements, official train+valid for final; target fit rows only for solver and preprocessing',
        test_labels_used=False,external_data=False,new_pretrained_weights=False,
        limitations='Historical raw proxies are not exact complete v86 OOF. Reused labels and conditional other assays; local gains do not estimate platform points.'))
    records=[]
    for k,task in enumerate(d['tasks']):
        metric=d['metrics'][k];bases={part:baseline(d,k,part) for part in ('fold0','fold1')}
        for cfg in CONFIGS:
            pp={part:expert(d,k,part,cfg) for part in bases}
            for w in WEIGHTS:
                parts=[]
                for part,base in bases.items():
                    q=r83.split(d,k,part)[1];pred=mix(metric,base,pp[part],w)
                    parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],base),
                        candidate=r83.score(metric,d['y'][q],pred),gain=r83.gain(metric,d['y'][q],base,pred)))
                records.append(dict(task=task,metric=metric,config=cfg,weight=w,parts=parts,objective=min(r['gain'] for r in parts)))
        best=max((r for r in records if r['task']==task),key=lambda r:r['objective']);emit(stage='development',task=task,best=best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=signature(),records=records))
    selected=[]
    for k,task in enumerate(d['tasks']):
        best=max((r for r in records if r['task']==task),key=lambda r:r['objective'])
        threshold=.003 if best['metric']=='MAE' else .001 if best['metric']=='Spearman' else .0005
        if best['objective']>threshold:selected.append(best)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=signature(),records=records))
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=signature(),selected=selected,confirmation_inspected=False))
    emit(stage='frozen',selected=selected)


def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==signature();records=[]
    for chosen in frozen['selected']:
        k=d['tasks'].index(chosen['task']);parts=[];metric=chosen['metric']
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];base=baseline(d,k,part);pred=mix(metric,base,expert(d,k,part,chosen['config']),chosen['weight'])
            parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],base),
                candidate=r83.score(metric,d['y'][q],pred),gain=r83.gain(metric,d['y'][q],base,pred)))
        r=dict(**chosen,confirmation=parts,accepted=all(p['gain']>0 for p in parts));records.append(r);emit(stage='confirmation',**r)
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=signature(),records=records,frozen_sha256=b.sha(OUT/'frozen_selection.json')))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=signature(),records=records,
        accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','confirm']);args=ap.parse_args();d=load()
    (develop if args.stage=='develop' else confirm)(d)
