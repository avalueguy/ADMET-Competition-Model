"""Separate conditional assay experts; no target assay in any input.

Supersedes invalid v94 attempt1. Each network fits its own endpoint only.
Perturbing every target query label must leave all training inputs unchanged.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v93_balanced_multitask as prev
b,r83=prev.b,prev.r83
OUT=ROOT/'admet_model_outputs_v94';ANCHOR,ANCHOR_SHA=prev.ANCHOR,prev.ANCHOR_SHA
SEEDS=(944126,944127);STEPS=1000;SNAPSHOTS=(100,300,1000);BATCH=128
MODES=('direct','residual');WEIGHTS=(.25,.5,1.)
def sig():return hashlib.sha256((b.sha(__file__)+prev.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def load():return prev.load()
def base(d,k,part):return prev.base(d,k,part)
def threshold(m):return prev.threshold(m)
def mix(m,old,p,w):return prev.mix(m,old,p,w)

def prepare(d,k,part):
    fit,q=r83.split(d,k,part);rows=np.r_[fit,q];raw=d['views']['physchem'].astype(float);mm=np.unique(d['mi'][fit])
    median=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in raw[mm].T]);full=np.where(np.isfinite(raw),raw,median);mean=full[mm].mean(0);std=full[mm].std(0).clip(.05)
    desc=((full[d['mi'][rows]]-mean)/std).clip(-5,5).astype(np.float32);fp=np.sqrt(d['views']['multiview'][d['mi'][rows],:12288]).astype(np.float32)
    memo=d.setdefault('_v94_aux',{});key=(k,part=='final')
    if key not in memo:memo[key]=r83.aux(d,k,part=='final')
    aux,cols,source,counts=memo[key];assert k not in cols and np.all(d['tid'][source]!=k) and np.all(d['split'][source]<(2 if part=='final' else 1))
    vals=aux[fit,:len(cols)];ac=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in vals.T]);sc=np.array([np.std(c[np.isfinite(c)]) if np.isfinite(c).any() else 1. for c in vals.T]).clip(.05)
    mask=aux[rows,len(cols):].copy();values=np.nan_to_num((aux[rows,:len(cols)]-ac)/sc).clip(-5,5)*mask;assay=np.c_[values,mask].astype(np.float32)
    yy=d['y'][fit];bp=r83.raw_proxy(d,fit);original=prev.prev.prev.prev.r88.r84.ORIGINAL
    qp=np.array([float(r['y_pred']) for r in b.table(original) if r['task']==d['tasks'][k]]) if part=='final' else r83.raw_proxy(d,q);metric=d['metrics'][k]
    if metric in ('AUROC','AUPRC'):
        beta=r83.calibrate(yy,r83.ranklog(bp));offset=beta[0]+beta[1]*r83.ranklog(bp);qoff=beta[0]+beta[1]*r83.ranklog(qp);target=np.c_[yy,yy];state=dict(center=0.,scale=1.,beta=beta.tolist())
    elif metric=='MAE':
        center=float(np.median(yy));scale=max(float(np.quantile(yy,.75)-np.quantile(yy,.25)),.001);target=np.c_[(yy-center)/scale,(yy-bp)/scale];offset=np.zeros(len(fit));qoff=qp;state=dict(center=center,scale=scale)
    else:
        z=r83.ranknorm(bp);x=np.c_[np.ones(len(z)),z];beta=np.linalg.solve(x.T@x+np.diag([1e-8,1.]),x.T@r83.ranknorm(yy));beta[1]=max(beta[1],.01);y=r83.ranknorm(yy);target=np.c_[y,y-x@beta];offset=np.zeros(len(fit));qoff=beta[0]+beta[1]*r83.ranknorm(qp);state=dict(center=0.,scale=1.,beta=beta.tolist())
    return dict(fit=fit,query=q,fp=fp,desc=desc,assay=assay,median=median,mean=mean,std=std,assay_center=ac,assay_scale=sc,cols=np.array(cols),source=source,counts=counts[q],target=target.astype(np.float32),offset=offset.astype(np.float32),qoffset=qoff,state=state)

class Net(nn.Module):
    def __init__(self,desc_dim,assay_dim):
        super().__init__();self.fp=nn.Sequential(nn.Linear(12288,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.4));self.desc=nn.Sequential(nn.Linear(desc_dim,32),nn.LayerNorm(32),nn.SiLU());self.assay=nn.Sequential(nn.Linear(assay_dim,48),nn.SiLU(),nn.Dropout(.25));self.head=nn.Sequential(nn.Linear(176,64),nn.SiLU(),nn.Dropout(.4),nn.Linear(64,2));nn.init.normal_(self.head[-1].weight,std=.005);nn.init.zeros_(self.head[-1].bias)
    def forward(self,fp,desc,assay):
        if self.training:
            n=assay.shape[1]//2;keep=(torch.rand_like(assay[:,:n])>.25).to(assay.dtype);assay=assay*torch.cat([keep,keep],1)
        return self.head(torch.cat([self.fp(fp),self.desc(desc),self.assay(assay)],1))

def tensors(a,device='cuda'):return tuple(torch.as_tensor(a[key],device=device) for key in ('fp','desc','assay'))
def infer(net,a,ts,batch=347):
    net.eval();start=len(a['fit']);out=[]
    with torch.inference_mode():
        for lo in range(start,len(a['fp']),batch):out.append(net(*(x[lo:lo+batch] for x in ts)).cpu().numpy())
    return np.concatenate(out).astype(float)

def train(d,k,part,seed):
    folder=OUT/'models'/d['tasks'][k]/part/str(seed);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);meta=json.loads((folder/'metadata.json').read_text());assert str(z['signature'])==meta['signature']==sig() and b.sha(file)==meta['predictions_sha256'];np.testing.assert_array_equal(z['query'],r83.split(d,k,part)[1]);return z['values'].copy()
    folder.mkdir(parents=True,exist_ok=True);started=time.monotonic();a=prepare(d,k,part);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);ts=tensors(a);target=torch.from_numpy(a['target']).cuda();offset=torch.from_numpy(a['offset']).cuda();net=Net(a['desc'].shape[1],a['assay'].shape[1]).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.05);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00005);snapshots=[];hashes={};metric=d['metrics'][k]
    for step in range(1,STEPS+1):
        net.train();ix=rng.choice(len(a['fit']),BATCH,replace=len(a['fit'])<BATCH);p=net(*(x[ix] for x in ts));yt=target[ix]
        if metric in ('AUROC','AUPRC'):loss=nn.functional.binary_cross_entropy_with_logits(p+torch.stack([torch.zeros_like(offset[ix]),offset[ix]],1),yt)
        elif metric=='MAE':loss=(p-yt).abs().mean()
        else:loss=(p-yt).square().mean()
        loss=loss+.01*p.square().mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();sched.step()
        if step in SNAPSHOTS:
            snapshots.append(infer(net,a,ts));path=folder/f'checkpoint_{step}.pt';torch.save(dict(signature=sig(),weights={name:value.detach().cpu().clone() for name,value in net.state_dict().items()},desc_dim=a['desc'].shape[1],assay_dim=a['assay'].shape[1],fit=a['fit'],query=a['query'],seed=seed,step=step,state=a['state']),path);hashes[path.name]=b.sha(path)
    values=np.stack(snapshots);np.savez_compressed(file,signature=sig(),fit=a['fit'],query=a['query'],values=values,snapshots=SNAPSHOTS,qoffset=a['qoffset']);b.save(folder/'metadata.json',dict(signature=sig(),part=part,task=d['tasks'][k],seed=seed,checkpoint_hashes=hashes,predictions_sha256=b.sha(file),seconds=time.monotonic()-started));emit(stage='fit',task=d['tasks'][k],part=part,seed=seed,seconds=time.monotonic()-started);del net,opt,ts,target,offset;torch.cuda.empty_cache();return values

def expert(d,k,part,cfg):
    memo=d.setdefault('_v94_experts',{});key=(k,part)
    if key not in memo:
        values=np.mean([train(d,k,part,s) for s in SEEDS],axis=0);a=prepare(d,k,part);memo[key]=(values,a['state'],a['qoffset'])
    values,s,qoff=memo[key];mode=MODES.index(cfg['mode']);raw=values[SNAPSHOTS.index(cfg['step']),:,mode]*s['scale']+(s['center'] if mode==0 else qoff);p=expit(raw) if d['metrics'][k] in ('AUROC','AUPRC') else raw
    return p.clip(0,100) if d['tasks'][k]=='ppbr_az' else p

def smoke(d):
    OUT.mkdir(exist_ok=True);k=d['tasks'].index('hia_hou');a=prepare(d,k,'fold0');q=a['query'];assert np.all(d['tid'][a['source']]!=k);ts=tensors(a);net=Net(a['desc'].shape[1],a['assay'].shape[1]).cuda().eval();p=infer(net,a,ts);np.testing.assert_allclose(p,infer(net,a,ts,113),atol=1e-6,rtol=1e-5)
    saved=d['y'][q].copy();d['y'][q]=1e6;d.pop('_v94_aux',None);z=prepare(d,k,'fold0');d['y'][q]=saved;d.pop('_v94_aux',None)
    for key in ('fp','desc','assay','target','offset','qoffset','fit','query','cols','source'):np.testing.assert_array_equal(a[key],z[key])
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),target_query_label_perturbation_invariance=True,target_assay_source_absent=True,alternate_batch=True));emit(stage='smoke',status='PASS')

def develop(d):
    OUT.mkdir(exist_ok=True);b.save(OUT/'protocol.json',dict(version=94,attempt=2,signature=sig(),baseline='v92',anchor_sha256=ANCHOR_SHA,baseline_official_score_user_reported=50.915,seeds=SEEDS,steps=STEPS,snapshots=SNAPSHOTS,modes=MODES,weights=WEIGHTS,selection_parts=['fold0','fold1'],confirmation_parts=['fold2','valid'],selection='Max minimum gain >1% MAE, .005 Spearman, .002 AUROC/AP; one config frozen per task.',confirmation='All four positive and mean exceeds threshold; no fallback.',architecture='Separate target-specific conditional neural experts. No shared loss or auxiliary source from target assay.',conditional=True,invalid_previous_attempt='Shared task loss allowed held target labels as other-task covariates. Discarded before final predictions.',test_labels_used=False,external_data=False,new_pretrained_weights=False,limitations='Other official assays are conditional inputs. Historical proxies not exact v92 OOF; labels reused. No official points estimate.'))
    b.save(OUT/'data_audit.json',dict(input_hashes=d['sources'],test_labels_finite=0));records=[];selected=[]
    for k,t in enumerate(d['tasks']):
        metric=d['metrics'][k];bases={p:base(d,k,p) for p in ('fold0','fold1')}
        for step in SNAPSHOTS:
            for mode in MODES:
                cfg=dict(step=step,mode=mode);ps={part:expert(d,k,part,cfg) for part in bases}
                for w in WEIGHTS:
                    parts=[]
                    for part,old in bases.items():
                        q=r83.split(d,k,part)[1];p=mix(metric,old,ps[part],w);parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],old),candidate=r83.score(metric,d['y'][q],p),gain=r83.gain(metric,d['y'][q],old,p)))
                    records.append(dict(task=t,metric=metric,config=cfg,weight=w,parts=parts,objective=min(x['gain'] for x in parts)))
        best=max((r for r in records if r['task']==t),key=lambda r:r['objective']);emit(stage='development',**best)
        if best['objective']>threshold(metric):selected.append(best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records));b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected,confirmation_inspected=False));emit(stage='frozen',selected=selected)

def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==sig();records=[]
    for ch in frozen['selected']:
        k=d['tasks'].index(ch['task']);m=ch['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];old=base(d,k,part);p=mix(m,old,expert(d,k,part,ch['config']),ch['weight']);parts.append(dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],old),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],old,p)))
        r=dict(**ch,confirmation=parts,accepted=all(x['gain']>0 for x in parts) and np.mean([x['gain'] for x in parts])>threshold(m));records.append(r);emit(stage='confirmation',**r)
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','develop','confirm']);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');dict(smoke=smoke,develop=develop,confirm=confirm)[a.stage](load())
