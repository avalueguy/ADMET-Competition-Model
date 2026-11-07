"""Fit-only standardized conformer/chemical kernels, exact GPU ridge solve."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v98_shape_experts as shape
import v92_full_kernel as historical
b,r83=shape.b,shape.r83
OUT=ROOT/'admet_model_outputs_v100';ANCHOR,ANCHOR_SHA=shape.ANCHOR,shape.ANCHOR_SHA
ALPHAS=(.1,1.,10.);WEIGHTS=(.25,.5,1.)
CONFIGS=[dict(name=f'{view}_{mode}_a{a}',view=view,mode=mode,alpha=a) for view in ('sum','product') for mode in ('direct','residual') for a in ALPHAS]

def sig():return hashlib.sha256((b.sha(__file__)+shape.sig()+historical.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def load():return shape.load()
def base(d,k,part):return shape.base(d,k,part)
def threshold(m):return shape.threshold(m)
def mix(m,old,p,w):return shape.mix(m,old,p,w)
def state(d,k,part):return historical.state(d,k,part)

def normalized(raw,fit,query):
    tr=raw[fit].astype(float);qq=raw[query].astype(float)
    center=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in tr.T])
    z=np.where(np.isfinite(tr),tr,center);std=z.std(0);active=np.flatnonzero(std>1e-5)
    std=std.clip(.05)
    a=((z[:,active]-center[active])/std[active]).clip(-6,6)
    q=((np.where(np.isfinite(qq),qq,center)[:,active]-center[active])/std[active]).clip(-6,6)
    return a.astype(np.float32),q.astype(np.float32),center,std,active

def matrices(d,k,part,smoke=False):
    fit,q=r83.split(d,k,part)
    if smoke:fit,q=fit[:70],q[:13]
    fi,qi=d['mi'][fit],d['mi'][q];n=len(fi);gr=torch.zeros((n,n),device='cuda');gq=torch.zeros((len(qi),n),device='cuda')
    raw=d['views']['multiview'][:,:12288]
    for start in range(0,12288,2048):
        tr=torch.as_tensor((raw[fi,start:start+2048]>0).astype(np.float32),device='cuda');tn=tr.sum(1)
        for ix,dest in ((fi,gr),(qi,gq)):
            for lo in range(0,len(ix),512):
                x=torch.as_tensor((raw[ix[lo:lo+512],start:start+2048]>0).astype(np.float32),device='cuda');dot=x@tr.T
                tan=(dot/(x.sum(1)[:,None]+tn[None]-dot).clamp_min(1e-8)).clamp(0,1)
                dest[lo:lo+len(x)]+=(.5*tan+.5*tan.square())/6
    sr=torch.zeros_like(gr);sq=torch.zeros_like(gq);states=[]
    for raw in (d['views']['physchem'],d['conformer_features']):
        tr,qe,center,std,active=normalized(raw,fi,qi);states.append(dict(center=center,scale=std,active=active))
        if not len(active):continue
        tx=torch.as_tensor(tr,device='cuda');tn=tx.square().sum(1)
        for source,dest in ((tr,sr),(qe,sq)):
            for lo in range(0,len(source),512):
                x=torch.as_tensor(source[lo:lo+512],device='cuda')
                dist=(x.square().sum(1)[:,None]+tn[None]-2*x@tx.T).clamp_min(0)/len(active)
                dest[lo:lo+len(x)]+=(.5*torch.exp(-.5*dist)+.5*torch.exp(-2*dist))/2
    sr.diagonal().fill_(1.)
    tr={'sum':.5*gr+.5*sr,'product':gr*sr};qe={'sum':.5*gq+.5*sq,'product':gq*sq}
    for key in tr:tr[key]=(tr[key]+tr[key].T)*.5
    return tr,qe,states

def fit_family(d,k,part):
    folder=OUT/'models'/d['tasks'][k]/part;file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);m=json.loads((folder/'metadata.json').read_text());assert str(z['signature'])==m['signature']==sig() and b.sha(file)==m['sha256']
        for key,ix in zip(('fit','query'),r83.split(d,k,part)):np.testing.assert_array_equal(z[key],ix)
        return dict(z)
    started=time.monotonic();a=state(d,k,part);tr,qe,states=matrices(d,k,part);rhs=torch.as_tensor(a['target'],dtype=torch.float32,device='cuda');values=[];coefs=[];errors=[]
    for view in ('sum','product'):
        pv=[];cv=[];er=[]
        for alpha in ALPHAS:
            mat=tr[view].clone();mat.diagonal().add_(alpha);chol=torch.linalg.cholesky(mat)
            coef=torch.cholesky_solve(rhs,chol);error=float((mat@coef-rhs).abs().max());assert error<.003,(view,alpha,error)
            pv.append((qe[view]@coef).cpu().numpy());cv.append(coef.cpu().numpy());er.append(error)
        values.append(pv);coefs.append(cv);errors.append(er)
    out=dict(signature=np.array(sig()),**a,values=np.array(values),coefficients=np.array(coefs),alphas=np.array(ALPHAS),solve_max_errors=np.array(errors))
    for j,s in enumerate(states):
        for key,val in s.items():out[f'view{j}_{key}']=val
    folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(file,**out)
    b.save(folder/'metadata.json',dict(signature=sig(),sha256=b.sha(file),task=d['tasks'][k],part=part,seconds=time.monotonic()-started,source='Own task fit labels; fit-only descriptor moments; chemical fingerprints; deterministic 3D descriptors.'))
    emit(stage='fit',task=d['tasks'][k],part=part,seconds=round(time.monotonic()-started));del tr,qe,rhs;torch.cuda.empty_cache();return out

def expert(d,k,part,cfg):
    memo=d.setdefault('_v100',{});key=(k,part)
    if key not in memo:memo[key]=fit_family(d,k,part)
    a=memo[key];m=d['metrics'][k];value=a['values'][('sum','product').index(cfg['view']),ALPHAS.index(cfg['alpha']),:,('direct','residual').index(cfg['mode'])]
    if cfg['mode']=='direct':p=value*float(a['scale'])+float(a['center'])
    elif m in ('AUROC','AUPRC'):p=expit(a['qoff'])+value
    else:p=a['qoff']+value*float(a['scale'])
    p=expit(4*(p-.5)) if m in ('AUROC','AUPRC') else p
    return p.clip(0,100) if d['tasks'][k]=='ppbr_az' else p

def smoke(d):
    OUT.mkdir(exist_ok=True);k=d['tasks'].index('cyp2c9_substrate_carbonmangels');tr,qe,_=matrices(d,k,'fold0',True)
    for view in tr:
        assert torch.linalg.eigvalsh(tr[view]).min()>-1e-5
        mat=tr[view]+.1*torch.eye(len(tr[view]),device='cuda');rhs=torch.arange(len(mat),device='cuda',dtype=torch.float32)[:,None]/100
        coef=torch.cholesky_solve(rhs,torch.linalg.cholesky(mat));ref=np.linalg.solve(mat.cpu().double().numpy(),rhs.cpu().double().numpy())
        np.testing.assert_allclose(coef.cpu().numpy(),ref,atol=3e-6,rtol=2e-5)
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),positive_semidefinite=True,cpu_solve=True));emit(stage='smoke',status='PASS')

def develop(d):
    OUT.mkdir(exist_ok=True);b.save(OUT/'protocol.json',dict(version=100,signature=sig(),baseline='v94',anchor_sha256=ANCHOR_SHA,configs=CONFIGS,weights=WEIGHTS,
        descriptor_preprocessing='Fit-only median imputation, std>1e-5, std>=.05, clip+-6; per-view squared distance divided by active dimensions.',
        kernel='Equal 2D/3D RBF views, each .5 exp(-.5d)+.5 exp(-2d). Sum or product with multi-fingerprint Tanimoto. Exact ridge .1/1/10.',
        selection='Max min fold0/fold1 >1% relative MAE/.005 Spearman/.002 AUC or AP.',confirmation='All four gains positive and mean above threshold; no fallback.',
        test_labels_used=False,external_data=False,new_pretrained_weights=False))
    records=[];selected=[]
    order=sorted(range(len(d['tasks'])),key=lambda k:int(((d['tid']==k)&(d['split']==0)).sum()))
    for k in order:
        m=d['metrics'][k];bases={p:base(d,k,p) for p in ('fold0','fold1')}
        for cfg in CONFIGS:
            pp={p:expert(d,k,p,cfg) for p in bases}
            for w in WEIGHTS:
                parts=[]
                for part,old in bases.items():
                    q=r83.split(d,k,part)[1];p=mix(m,old,pp[part],w)
                    parts.append(dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],old),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],old,p)))
                records.append(dict(task=d['tasks'][k],metric=m,config=cfg,weight=w,parts=parts,objective=min(x['gain'] for x in parts)))
        best=max((r for r in records if r['task']==d['tasks'][k]),key=lambda r:r['objective']);emit(stage='development',**best)
        if best['objective']>threshold(m):selected.append(best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records));b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected,confirmation_inspected=False));emit(stage='frozen',selected=selected)

def confirm(d):
    f=json.loads((OUT/'frozen_selection.json').read_text());assert f['signature']==sig();records=[]
    for ch in f['selected']:
        k=d['tasks'].index(ch['task']);m=ch['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];old=base(d,k,part);p=mix(m,old,expert(d,k,part,ch['config']),ch['weight'])
            parts.append(dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],old),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],old,p)))
        r=dict(**ch,confirmation=parts,accepted=bool(all(x['gain']>0 for x in parts) and np.mean([x['gain'] for x in parts])>threshold(m)));records.append(r);emit(stage='confirmation',**r)
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','develop','confirm']);args=ap.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](load())
