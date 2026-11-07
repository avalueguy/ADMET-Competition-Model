"""Fit-only local chemical residual interpolation against fixed v88.

No task auxiliary labels: neighbors are target-task fit rows only. Exact
canonical matches are excluded. Local Newton residuals for classification,
weighted median residuals for MAE, local rank residuals for Spearman.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v90_broad_residual as prev
b,r83=prev.b,prev.r83
OUT=ROOT/'admet_model_outputs_v91'
ANCHOR,ANCHOR_SHA=prev.ANCHOR,prev.ANCHOR_SHA
CONFIGS=[dict(name=f'{v}_k{k}_p{p}_a{a}',view=v,neighbors=k,power=p,alpha=a)
    for v in ('morgan','multi') for k in (16,64) for p in (4,16) for a in (1.,10.)]
WEIGHTS=(.25,.5,1.)
def sig():return hashlib.sha256((b.sha(__file__)+prev.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def load():return prev.load()
def base(d,k,part):return prev.base(d,k,part)
def mix(metric,old,p,w):return prev.r88.mix(metric,old,p,w)

def state(d,k,part):
    fit,q=r83.split(d,k,part);metric=d['metrics'][k];basefit=r83.raw_proxy(d,fit)
    qp=(np.array([float(r['y_pred']) for r in b.table(prev.r88.r84.ORIGINAL) if r['task']==d['tasks'][k]]) if part=='final' else r83.raw_proxy(d,q))
    y=d['y'][fit]
    if metric in ('AUROC','AUPRC'):
        z,qz=r83.ranklog(basefit),r83.ranklog(qp);beta=r83.calibrate(y,z)
        off,qoff=beta[0]+beta[1]*z,beta[0]+beta[1]*qz
        pr=expit(off);residual=y-pr;curvature=(pr*(1-pr)).clip(.03);scale=1.
    elif metric=='MAE':
        scale=max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-3)
        residual=(y-basefit)/scale;curvature=np.ones(len(fit));qoff=qp
    else:
        z,qz=r83.ranknorm(basefit),r83.ranknorm(qp);x=np.c_[np.ones(len(z)),z]
        beta=np.linalg.solve(x.T@x+np.diag([1e-8,1.]),x.T@r83.ranknorm(y));beta[1]=max(beta[1],.01)
        residual=r83.ranknorm(y)-x@beta;curvature=np.ones(len(fit));qoff=beta[0]+beta[1]*qz;scale=1.
    return dict(fit=fit,query=q,base_fit=basefit,base_query=qp,residual=residual,curvature=curvature,qoff=qoff,scale=np.array(scale))

def neighbors(d,k,part,view):
    folder=OUT/'neighbors'/d['tasks'][k]/part;path=folder/(view+'.npz')
    if path.exists():
        zz=np.load(path);assert str(zz['signature'])==sig();np.testing.assert_array_equal(zz['query'],r83.split(d,k,part)[1]);return dict(zz)
    started=time.monotonic();a=state(d,k,part);fit,q=a['fit'],a['query']
    unique,inverse=np.unique(d['canonical'][fit],return_inverse=True)
    first=np.array([np.flatnonzero(inverse==j)[0] for j in range(len(unique))]);refrows=fit[first]
    rr=np.array([np.median(a['residual'][inverse==j]) for j in range(len(unique))])
    hh=np.array([np.median(a['curvature'][inverse==j]) for j in range(len(unique))])
    fp=d['fingerprint'] if view=='morgan' else d['views']['multiview'][:,:-435]
    assert np.min(fp)>=0
    take=min(128,len(refrows));scores=[];indices=[]
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    gx=torch.as_tensor(fp,device='cuda');norm=gx.square().sum(1);reference=gx[d['mi'][refrows]];rn=reference.square().sum(1)
    for start in range(0,len(q),256):
        rows=q[start:start+256];xx=gx[d['mi'][rows]];dot=xx@reference.T
        sim=(dot/(norm[d['mi'][rows],None]+rn[None]-dot).clamp_min(1e-8)).clamp(0,1)
        # Exact molecule labels in fit are never copied to a query molecule.
        exact=d['canonical'][rows,None]==unique[None,:];sim[torch.as_tensor(exact,device='cuda')]=0.
        top,ix=torch.topk(sim,take,dim=1,sorted=True);scores.append(top.cpu().numpy());indices.append(ix.cpu().numpy())
    out=dict(signature=np.array(sig()),**a,refrows=refrows,refs=unique,reference_residual=rr,reference_curvature=hh,
        similarity=np.concatenate(scores),neighbor_positions=np.concatenate(indices))
    folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**out)
    b.save(path.with_suffix('.json'),dict(signature=sig(),sha256=b.sha(path),task=d['tasks'][k],part=part,view=view,
        target_fit_only=True,exact_canonical_matches_excluded=True,seconds=time.monotonic()-started))
    del gx,reference;torch.cuda.empty_cache();return out

def infer(a,metric,cfg):
    take=min(cfg['neighbors'],a['similarity'].shape[1]);sim=a['similarity'][:,:take].astype(float);ix=a['neighbor_positions'][:,:take]
    w=sim**cfg['power'];r=a['reference_residual'][ix];h=a['reference_curvature'][ix]
    if metric=='MAE':
        order=np.argsort(r,axis=1);sr=np.take_along_axis(r,order,axis=1);sw=np.take_along_axis(w,order,axis=1)
        total=w.sum(1);pos=(np.cumsum(sw,axis=1)<(.5*total[:,None])).sum(1).clip(0,take-1)
        correction=sr[np.arange(len(r)),pos]*total/(total+cfg['alpha'])
    else:correction=(w*r).sum(1)/((w*h).sum(1)+cfg['alpha'])
    raw=a['qoff']+correction*float(a['scale'])
    return expit(raw) if metric in ('AUROC','AUPRC') else raw

def expert(d,k,part,cfg):
    p=infer(neighbors(d,k,part,cfg['view']),d['metrics'][k],cfg)
    return p.clip(0,100) if d['tasks'][k]=='ppbr_az' else p

def develop(d):
    OUT.mkdir(exist_ok=True)
    b.save(OUT/'protocol.json',dict(version=91,signature=sig(),baseline='v88',anchor_sha256=ANCHOR_SHA,
        baseline_official_score_user_reported=50.910,configs=CONFIGS,weights=WEIGHTS,
        selection_parts=['fold0','fold1'],confirmation_parts=['fold2','valid'],
        selection='Max minimum gain; >.003 relative MAE, .001 Spearman, .0005 AUROC/AUPRC; one fixed task/config/weight.',
        confirmation='All four positive; no fallback tuning.',new_direction='Local target-assay residual smoothing with molecular neighbors; no auxiliary assays',
        exact_canonical_matches_excluded=True,neighbors_from_target_fit_only=True,test_labels_used=False,external_data=False,new_pretrained_weights=False,
        limitations='Historical proxy not full v88 OOF; historical labels reused; no official point estimate.'))
    b.save(OUT/'data_audit.json',dict(input_hashes=d['sources'],legacy_folds_sha256=b.sha(ROOT/'recovered_history/legacy_v24_folds.npy'),test_labels_finite=0))
    records=[]
    for k,task in enumerate(d['tasks']):
        metric=d['metrics'][k];bases={part:base(d,k,part) for part in ('fold0','fold1')}
        for cfg in CONFIGS:
            pp={part:expert(d,k,part,cfg) for part in bases}
            for w in WEIGHTS:
                parts=[]
                for part,old in bases.items():
                    q=r83.split(d,k,part)[1];p=mix(metric,old,pp[part],w)
                    parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],old),candidate=r83.score(metric,d['y'][q],p),gain=r83.gain(metric,d['y'][q],old,p)))
                records.append(dict(task=task,metric=metric,config=cfg,weight=w,parts=parts,objective=min(r['gain'] for r in parts)))
        best=max((r for r in records if r['task']==task),key=lambda r:r['objective']);emit(stage='development',**best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    selected=[]
    for task in d['tasks']:
        best=max((r for r in records if r['task']==task),key=lambda r:r['objective']);threshold=.003 if best['metric']=='MAE' else .001 if best['metric']=='Spearman' else .0005
        if best['objective']>threshold:selected.append(best)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records))
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected,confirmation_inspected=False));emit(stage='frozen',selected=selected)

def confirm(d):
    f=json.loads((OUT/'frozen_selection.json').read_text());assert f['signature']==sig();records=[]
    for ch in f['selected']:
        k=d['tasks'].index(ch['task']);metric=ch['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];old=base(d,k,part);p=mix(metric,old,expert(d,k,part,ch['config']),ch['weight'])
            parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],old),candidate=r83.score(metric,d['y'][q],p),gain=r83.gain(metric,d['y'][q],old,p)))
        rec=dict(**ch,confirmation=parts,accepted=all(r['gain']>0 for r in parts));records.append(rec);emit(stage='confirmation',**rec)
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records,frozen_sha256=b.sha(OUT/'frozen_selection.json')))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))

def smoke(d):
    k=d['tasks'].index('cyp2c9_substrate_carbonmangels');a=neighbors(d,k,'fold0','morgan');cfg=CONFIGS[0];p=infer(a,d['metrics'][k],cfg)
    ids=np.unique([0,len(p)//2,len(p)-1]); sub=dict(a)
    for key in ('qoff','similarity','neighbor_positions'):sub[key]=a[key][ids]
    np.testing.assert_allclose(p[ids],infer(sub,d['metrics'][k],cfg),atol=1e-12)
    assert np.all(d['split'][a['refrows']]==0) and np.all(d['tid'][a['refrows']]==k)
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),subset_prediction='PASS',reference_fit_only=True));emit(stage='smoke',status='PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','develop','confirm']);args=ap.parse_args();dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](load())
