"""INVALID ATTEMPT: retained for audit only; use v94_isolated_assay.py.

Official-only leave-target-out assay imputation with chemical context.

All official train measurements from other endpoints can be covariates.
Target value and mask are zeroed on every training/inference example.
Scaffold-held target labels are excluded from loss, not exposed as input.
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
SEEDS=(944026,944027)
STEPS=5000
SNAPSHOTS=(500,1500,5000)
PER_TASK=32
MODES=('direct','residual');WEIGHTS=(.25,.5,1.)
def sig():return hashlib.sha256((b.sha(__file__)+prev.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def load():return prev.load()
def base(d,k,part):return prev.base(d,k,part)
def threshold(m):return prev.threshold(m)
def mix(m,old,p,w):return prev.mix(m,old,p,w)
def split(d,part):return prev.split(d,part)

def prepare(d,part):
    a=prev.prepare(d,part);source=np.flatnonzero(d['split']<(2 if part=='final' else 1));tasks=len(d['tasks'])
    can,inverse=np.unique(d['canonical'],return_inverse=True);lookup={};full=np.zeros((len(can),tasks),np.float32);mask=np.zeros_like(full);center=[];scale=[]
    # Split target statistics use only loss-fitting rows, never query labels.
    for k in range(tasks):
        y=d['y'][a['members'][k]]
        center.append(0. if d['metrics'][k] in ('AUROC','AUPRC') else float(np.median(y)))
        scale.append(1. if d['metrics'][k] in ('AUROC','AUPRC') else max(float(np.quantile(y,.75)-np.quantile(y,.25)),.001))
    for i in source:lookup.setdefault((int(inverse[i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
    for (i,k),ys in lookup.items():full[i,k]=np.clip((np.median(ys)-center[k])/scale[k],-5,5);mask[i,k]=1.
    # Identity of a molecule only determines the official covariate lookup.
    a.update(assay=full,mask=mask,assay_row=inverse,source=source,assay_center=np.array(center),assay_scale=np.array(scale))
    return a

class Net(nn.Module):
    def __init__(self,desc_dim,tasks):
        super().__init__();self.tasks=tasks
        self.fp=nn.Sequential(nn.Linear(12288,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.3))
        self.desc=nn.Sequential(nn.Linear(desc_dim,48),nn.LayerNorm(48),nn.SiLU())
        self.assay=nn.Sequential(nn.Linear(tasks*2,96),nn.SiLU(),nn.Dropout(.25),nn.Linear(96,64),nn.SiLU())
        self.trunk=nn.Sequential(nn.Linear(240,160),nn.LayerNorm(160),nn.SiLU(),nn.Dropout(.35))
        self.head_weight=nn.Parameter(torch.empty(tasks,160,48));nn.init.xavier_uniform_(self.head_weight)
        self.head_bias=nn.Parameter(torch.zeros(tasks,48));self.out_weight=nn.Parameter(torch.empty(tasks,48,2));nn.init.normal_(self.out_weight,std=.005);self.out_bias=nn.Parameter(torch.zeros(tasks,2))
    def forward(self,fp,desc,values,mask,tid,drop=.25):
        keep=1.-nn.functional.one_hot(tid,self.tasks).to(values.dtype)
        if self.training:keep=keep*(torch.rand_like(mask)>drop)
        mask=mask*keep;values=values*mask
        h=self.trunk(torch.cat([self.fp(fp),self.desc(desc),self.assay(torch.cat([values,mask],1))],1))
        h=nn.functional.silu(torch.bmm(h[:,None],self.head_weight[tid]).squeeze(1)+self.head_bias[tid]);h=nn.functional.dropout(h,.3,self.training)
        return torch.bmm(h[:,None],self.out_weight[tid]).squeeze(1)+self.out_bias[tid]

def tensors(a,device='cuda'):
    return tuple(torch.as_tensor(a[key],device=device) for key in ('fp','desc','assay','mask'))

def infer(net,d,a,ts,batch=512):
    net.eval();q=a['query'];fp,desc,assay,mask=ts;out=[]
    with torch.inference_mode():
        for start in range(0,len(q),batch):
            ix=q[start:start+batch];mm=d['mi'][ix];rr=a['assay_row'][ix];tid=torch.as_tensor(d['tid'][ix],device=fp.device)
            out.append(net(fp[mm],desc[mm],assay[rr],mask[rr],tid).cpu().numpy())
    return np.concatenate(out).astype(float)

def train(d,part,seed):
    folder=OUT/'models'/part/str(seed);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);meta=json.loads((folder/'metadata.json').read_text());assert str(z['signature'])==meta['signature']==sig() and b.sha(file)==meta['predictions_sha256'];np.testing.assert_array_equal(z['query'],split(d,part)[1]);return z['values'].copy()
    started=time.monotonic();folder.mkdir(parents=True,exist_ok=True);a=prepare(d,part);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    ts=tensors(a);target=torch.from_numpy(a['target']).cuda();offset=torch.from_numpy(a['offset']).cuda();net=Net(a['desc'].shape[1],len(d['tasks'])).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.04);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00003)
    kinds=torch.as_tensor([0 if m in ('AUROC','AUPRC') else 1 if m=='MAE' else 2 for m in d['metrics']],device='cuda');snapshots=[];hashes={};losses=[];fp,desc,assay,mask=ts
    for step in range(1,STEPS+1):
        net.train();ix=np.concatenate([rng.choice(tr,PER_TASK,replace=len(tr)<PER_TASK) for tr in a['members']]);tid=torch.as_tensor(d['tid'][ix],device='cuda');mm=d['mi'][ix];rr=a['assay_row'][ix]
        out=net(fp[mm],desc[mm],assay[rr],mask[rr],tid,drop=.5 if step%2==0 else .1);yt=target[ix];off=offset[ix];kind=kinds[tid]
        cls=nn.functional.binary_cross_entropy_with_logits(out+torch.stack([torch.zeros_like(off),off],1),yt.clamp(0,1),reduction='none');loss=torch.where((kind==0)[:,None],cls,torch.where((kind==1)[:,None],(out-yt).abs(),(out-yt).square())).mean()+.01*out.square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();sched.step()
        if step%500==0:losses.append(dict(step=step,loss=float(loss.detach())));emit(stage='train',part=part,seed=seed,step=step,seconds=time.monotonic()-started,loss=float(loss.detach()))
        if step in SNAPSHOTS:
            snapshots.append(infer(net,d,a,ts));p=folder/f'checkpoint_{step}.pt';torch.save(dict(signature=sig(),state={k:x.detach().cpu().clone() for k,x in net.state_dict().items()},desc_dim=a['desc'].shape[1],tasks=len(d['tasks']),fit=a['fit'],query=a['query'],seed=seed,step=step,median=a['median'],mean=a['mean'],std=a['std'],states=a['states'],qoffset=a['qoffset'],source=a['source'],assay_center=a['assay_center'],assay_scale=a['assay_scale']),p);hashes[p.name]=b.sha(p)
    values=np.stack(snapshots);np.savez_compressed(file,signature=sig(),fit=a['fit'],query=a['query'],values=values,snapshots=SNAPSHOTS,qoffset=a['qoffset']);b.save(folder/'metadata.json',dict(signature=sig(),part=part,seed=seed,steps=STEPS,snapshots=SNAPSHOTS,checkpoint_hashes=hashes,predictions_sha256=b.sha(file),seconds=time.monotonic()-started,losses=losses));del net,opt,ts,fp,desc,assay,mask,target,offset;torch.cuda.empty_cache();return values

def expert(d,k,part,cfg):
    memo=d.setdefault('_v94_experts',{})
    if part not in memo:
        values=np.mean([train(d,part,s) for s in SEEDS],axis=0);f=OUT/'models'/part/str(SEEDS[0]);cache=dict(np.load(f/'predictions.npz'));ck=torch.load(f/f'checkpoint_{SNAPSHOTS[0]}.pt',map_location='cpu',weights_only=False);memo[part]=(values,cache['query'],cache['qoffset'],ck['states'])
    values,q,qoff,states=memo[part];j=SNAPSHOTS.index(cfg['step']);mode=MODES.index(cfg['mode']);active=d['tid'][q]==k;s=states[k]
    raw=values[j,active,mode]*s['scale']+(s['center'] if mode==0 else qoff[active]);p=expit(raw) if d['metrics'][k] in ('AUROC','AUPRC') else raw
    return p.clip(0,100) if d['tasks'][k]=='ppbr_az' else p

def smoke(d):
    OUT.mkdir(exist_ok=True);a=prepare(d,'fold0');q=a['query'][:41];ix=a['assay_row'][q];mm=d['mi'][q];tid=torch.as_tensor(d['tid'][q],device='cuda');ts=tensors(a);fp,desc,assay,mask=ts;net=Net(desc.shape[1],len(d['tasks'])).cuda().eval()
    with torch.inference_mode():
        x=net(fp[mm],desc[mm],assay[ix],mask[ix],tid);values=assay[ix].clone();m=mask[ix].clone();values[torch.arange(len(q)),tid]=1e6;m[torch.arange(len(q)),tid]=1.
        y=net(fp[mm],desc[mm],values,m,tid);torch.testing.assert_close(x,y,atol=0,rtol=0)
    aa=dict(a,query=q);np.testing.assert_allclose(infer(net,d,aa,ts),infer(net,d,aa,ts,13),atol=1e-6,rtol=1e-5)
    assert np.all(d['split'][a['source']]==0);b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),target_value_and_mask_invariance=True,alternate_batch=True,source_train_only=True,test_labels_finite=0));emit(stage='smoke',status='PASS')

def develop(d):
    OUT.mkdir(exist_ok=True);b.save(OUT/'protocol.json',dict(version=94,signature=sig(),baseline='v92',anchor_sha256=ANCHOR_SHA,baseline_official_score_user_reported=50.915,seeds=SEEDS,steps=STEPS,snapshots=SNAPSHOTS,modes=MODES,weights=WEIGHTS,batch_per_task=PER_TASK,selection_parts=['fold0','fold1'],confirmation_parts=['fold2','valid'],selection='Max minimum gain >1% MAE, .005 Spearman, .002 AUROC/AP; one config frozen per task.',confirmation='All four positive and mean exceeds threshold; no fallback.',architecture='Shared official assay denoising imputer with chemical structure, missing masks, target mask always removed and other-input dropout.',conditional=True,assay_source='Official train only until final; final train+valid. Target value and target mask always zero.',test_labels_used=False,external_data=False,new_pretrained_weights=False,limitations='Conditional same-molecule official measurements are inputs. Historical proxy not exact v92 OOF; labels reused. No estimate of official points.'))
    b.save(OUT/'data_audit.json',dict(input_hashes=d['sources'],test_labels_finite=0))
    for part in ('fold0','fold1'):
        for seed in SEEDS:train(d,part,seed)
    records=[];selected=[]
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
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records));b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected,confirmation_inspected=False));emit(stage='frozen',selected=selected)

def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==sig();records=[]
    if frozen['selected']:
        for part in ('fold2','valid'):
            for seed in SEEDS:train(d,part,seed)
    for ch in frozen['selected']:
        k=d['tasks'].index(ch['task']);m=ch['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];old=base(d,k,part);p=mix(m,old,expert(d,k,part,ch['config']),ch['weight']);parts.append(dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],old),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],old,p)))
        r=dict(**ch,confirmation=parts,accepted=all(x['gain']>0 for x in parts) and np.mean([x['gain'] for x in parts])>threshold(m));records.append(r);emit(stage='confirmation',**r)
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))

if __name__=='__main__':
    raise SystemExit('Invalid shared-assay input isolation. Use v94_isolated_assay.py; do not reuse these models.')
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','develop','confirm']);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');dict(smoke=smoke,develop=develop,confirm=confirm)[a.stage](load())
