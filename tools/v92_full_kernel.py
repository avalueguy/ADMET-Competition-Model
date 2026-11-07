"""Full-rank GPU Tanimoto ridge experts, direct and residual, fixed v88 control.

Uses every own-task fit molecule. No landmarks, auxiliary labels, or new
features/pretraining. The exact shared matrix is solved at four ridge levels.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v91_local_residual as prev
b,r83=prev.b,prev.r83
OUT=ROOT/'admet_model_outputs_v92';ANCHOR,ANCHOR_SHA=prev.ANCHOR,prev.ANCHOR_SHA
ALPHAS=(.03,.3,3.,30.)
CONFIGS=[dict(name=f'{v}_{m}_a{a}',view=v,mode=m,alpha=a) for v in ('ecfp','multi') for m in ('direct','residual') for a in ALPHAS]
WEIGHTS=(.25,.5,1.)
def sig():return hashlib.sha256((b.sha(__file__)+prev.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def load():return prev.load()
def base(d,k,part):return prev.base(d,k,part)
def mix(metric,old,p,w):return prev.mix(metric,old,p,w)

def state(d,k,part):
    a=prev.state(d,k,part);y=d['y'][a['fit']];metric=d['metrics'][k]
    if metric in ('AUROC','AUPRC'):
        center=float(y.mean());scale=1.;direct=y-center
        off=a['qoff'];residual=a['residual']
    elif metric=='MAE':
        center=float(np.median(y));scale=float(a['scale']);direct=(y-center)/scale;off=a['base_query'];residual=a['residual']
    else:
        center=0.;scale=1.;direct=r83.ranknorm(y);off=a['qoff'];residual=a['residual']
    return dict(fit=a['fit'],query=a['query'],target=np.c_[direct,residual],center=np.array(center),scale=np.array(scale),qoff=off)

def matrices(d,k,part,view,smoke=False):
    fit,q=r83.split(d,k,part)
    if smoke:fit,q=fit[:80],q[:11]
    raw=d['views']['multiview'][:,:-435];slices=[(0,2048),(2048,4096)] if view=='ecfp' else [(j,j+2048) for j in range(0,12288,2048)]
    train=torch.zeros((len(fit),len(fit)),device='cuda');query=torch.zeros((len(q),len(fit)),device='cuda')
    for lo,hi in slices:
        xx=torch.as_tensor((raw[:,lo:hi]>0).astype(np.float32),device='cuda');ref=xx[d['mi'][fit]];rn=ref.sum(1)
        for rows,out in ((fit,train),(q,query)):
            for start in range(0,len(rows),512):
                x=xx[d['mi'][rows[start:start+512]]];dot=x@ref.T
                tan=(dot/(x.sum(1)[:,None]+rn[None]-dot).clamp_min(1e-8)).clamp(0,1)
                out[start:start+len(x)]+=(.5*tan+.5*tan.square())/len(slices)
    train=(train+train.T)*.5
    return train,query

def fit_family(d,k,part,view):
    folder=OUT/'models'/d['tasks'][k]/part;path=folder/(view+'.npz')
    if path.exists():
        z=np.load(path);meta=json.loads(path.with_suffix('.json').read_text());assert str(z['signature'])==meta['signature']==sig() and b.sha(path)==meta['sha256']
        np.testing.assert_array_equal(z['fit'],r83.split(d,k,part)[0]);np.testing.assert_array_equal(z['query'],r83.split(d,k,part)[1]);return dict(z)
    started=time.monotonic();a=state(d,k,part);tr,qe=matrices(d,k,part,view)
    rhs=torch.as_tensor(a['target'],dtype=torch.float32,device='cuda');coefs=[];preds=[];errors=[]
    for alpha in ALPHAS:
        mat=tr.clone();mat.diagonal().add_(alpha)
        chol=torch.linalg.cholesky(mat);coef=torch.cholesky_solve(rhs,chol);pred=qe@coef
        error=(mat@coef-rhs).abs().max().item();assert error<.003,(alpha,error)
        coefs.append(coef.cpu().numpy());preds.append(pred.cpu().numpy());errors.append(error)
    out=dict(signature=np.array(sig()),**a,alphas=np.array(ALPHAS),coefficients=np.stack(coefs),values=np.stack(preds),solve_max_errors=np.array(errors))
    folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**out)
    b.save(path.with_suffix('.json'),dict(signature=sig(),sha256=b.sha(path),view=view,task=d['tasks'][k],part=part,
        fit_rows=len(a['fit']),query_rows=len(a['query']),seconds=time.monotonic()-started,full_rank=True,auxiliary_labels=False))
    del tr,qe,rhs;torch.cuda.empty_cache();emit(stage='fit',task=d['tasks'][k],part=part,view=view,seconds=time.monotonic()-started)
    return out

def infer(a,metric,cfg):
    j=ALPHAS.index(cfg['alpha']);mode=0 if cfg['mode']=='direct' else 1;value=a['values'][j,:,mode]
    if cfg['mode']=='direct':p=value*float(a['scale'])+float(a['center'])
    elif metric in ('AUROC','AUPRC'):
        prob=expit(a['qoff']);p=prob+value
    else:p=a['qoff']+value*float(a['scale'])
    # Smooth monotone map retains classification ranking and legal confidence range.
    return expit(4*(p-.5)) if metric in ('AUROC','AUPRC') else p

def expert(d,k,part,cfg):
    p=infer(fit_family(d,k,part,cfg['view']),d['metrics'][k],cfg)
    return p.clip(0,100) if d['tasks'][k]=='ppbr_az' else p

def develop(d):
    OUT.mkdir(exist_ok=True)
    b.save(OUT/'protocol.json',dict(version=92,signature=sig(),baseline='v88',anchor_sha256=ANCHOR_SHA,baseline_official_score_user_reported=50.910,
        configs=CONFIGS,weights=WEIGHTS,selection_parts=['fold0','fold1'],confirmation_parts=['fold2','valid'],
        selection='Max minimum gain; >.003 relative MAE, .001 Spearman, .0005 AUROC/AUPRC; freeze one task/config/weight.',
        confirmation='All four positive; no fallback tuning.',architecture='Full-rank binary multi-fingerprint Tanimoto ridge with direct and historical residual targets',
        training_labels='Target fit rows only',test_labels_used=False,external_data=False,new_pretrained_weights=False,
        limitations='Historical proxies not complete v88 OOF; labels previously reused. Local gain not official score.'))
    b.save(OUT/'data_audit.json',dict(input_hashes=d['sources'],test_labels_finite=0,legacy_folds_sha256=b.sha(ROOT/'recovered_history/legacy_v24_folds.npy')))
    records=[]
    order=sorted(range(len(d['tasks'])),key=lambda k:int(((d['tid']==k)&(d['split']==0)).sum()))
    for k in order:
        task,metric=d['tasks'][k],d['metrics'][k];bases={p:base(d,k,p) for p in ('fold0','fold1')}
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
    for k in order:
        best=max((r for r in records if r['task']==d['tasks'][k]),key=lambda r:r['objective']);threshold=.003 if best['metric']=='MAE' else .001 if best['metric']=='Spearman' else .0005
        if best['objective']>threshold:selected.append(best)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records));b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected,confirmation_inspected=False));emit(stage='frozen',selected=selected)

def confirm(d):
    f=json.loads((OUT/'frozen_selection.json').read_text());assert f['signature']==sig();records=[]
    for ch in f['selected']:
        k=d['tasks'].index(ch['task']);metric=ch['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];old=base(d,k,part);p=mix(metric,old,expert(d,k,part,ch['config']),ch['weight'])
            parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],old),candidate=r83.score(metric,d['y'][q],p),gain=r83.gain(metric,d['y'][q],old,p)))
        rec=dict(**ch,confirmation=parts,accepted=all(x['gain']>0 for x in parts));records.append(rec);emit(stage='confirmation',**rec)
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records,frozen_sha256=b.sha(OUT/'frozen_selection.json')))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))

def smoke(d):
    k=d['tasks'].index('cyp2c9_substrate_carbonmangels');tr,qe=matrices(d,k,'fold0','multi',True)
    rows,q=r83.split(d,k,'fold0');raw=d['views']['multiview'][:,:12288];cpu=np.zeros((len(q[:11]),len(rows[:80])))
    for lo in range(0,12288,2048):
        x=(raw[d['mi'][q[:11]],lo:lo+2048]>0).astype(float);r=(raw[d['mi'][rows[:80]],lo:lo+2048]>0).astype(float);dot=x@r.T;tan=dot/np.maximum(x.sum(1)[:,None]+r.sum(1)[None]-dot,1e-8);cpu+=(.5*tan+.5*tan**2)/6
    np.testing.assert_allclose(qe.cpu().numpy(),cpu,atol=1e-7)
    mat=tr+torch.eye(len(tr),device='cuda')*.3;rhs=torch.arange(len(tr),device='cuda',dtype=torch.float32)[:,None]/100
    coef=torch.cholesky_solve(rhs,torch.linalg.cholesky(mat));ref=np.linalg.solve(mat.cpu().double().numpy(),rhs.cpu().double().numpy())
    np.testing.assert_allclose(coef.cpu().numpy(),ref,atol=2e-6);OUT.mkdir(exist_ok=True);b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),cpu_kernel='PASS',cpu_solve='PASS'));emit(stage='smoke',status='PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','develop','confirm']);args=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](load())
