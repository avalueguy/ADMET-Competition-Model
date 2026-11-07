"""Shared molecule/task-gated residual neural experts with global exclusions.

Every outer query scaffold is removed from all tasks' training labels.
Fixed checkpoints are selected on fold0/1 only. No external pretraining.
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
from torch import nn
from scipy.special import expit
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import admet_v77_ranking as b
import v83_residual as r83
import v88_context_kernel as r88

OUT=ROOT/'admet_model_outputs_v89'
ANCHOR=ROOT/'admet_submissions/v88_context_kernel/我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
ANCHOR_SHA='b8b339db361ca11499300f998fa9418f3e947db751c1c3cd960936b59defd20d'
SEEDS=(894026,894027)
STEPS=2400
SNAPSHOTS=(1200,2400)
BATCH=256
WEIGHTS=(.25,.5,1.)


def signature():
    return hashlib.sha256(''.join(b.sha(p) for p in (__file__,b.__file__,r83.__file__,r88.__file__,r88.r87.__file__)).encode()).hexdigest()


def emit(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True)


def load():
    d=r88.load();assert b.sha(ANCHOR)==ANCHOR_SHA
    return d


def baseline(d,k,part):
    if part=='final':return np.array([float(r['y_pred']) for r in b.table(ANCHOR) if r['task']==d['tasks'][k]])
    if d['tasks'][k]=='cyp3a4_substrate_carbonmangels':
        c=json.loads((r88.OUT/'confirmation.json').read_text());r=next(r for r in c['records'] if r['task']==d['tasks'][k] and r['accepted'])
        return r88.mix(r['metric'],r88.baseline(d,k,part),r88.expert(d,k,part,r['config']),r['weight'])
    return r88.baseline(d,k,part)


def split(d,part):
    if part=='final':q=np.flatnonzero(d['split']==2);fit=np.flatnonzero(d['split']<2)
    else:
        q=np.flatnonzero(d['split']==1) if part=='valid' else np.flatnonzero((d['split']==0)&(d['legacy_folds']==int(part[-1])))
        fit=np.flatnonzero((d['split']==0)&~np.isin(d['groups'],d['groups'][q]))
        assert not set(d['groups'][fit])&set(d['groups'][q]);assert not set(d['canonical'][fit])&set(d['canonical'][q])
    assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    for k in range(len(d['tasks'])):np.testing.assert_array_equal(q[d['tid'][q]==k],r83.split(d,k,part)[1])
    return fit,q


def prepare(d,part):
    fit,q=split(d,part);mols=np.unique(d['mi'][fit]);raw=d['views']['physchem'].astype(float)
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in raw[mols].T])
    full=np.where(np.isfinite(raw),raw,med);mean=full[mols].mean(0);std=full[mols].std(0).clip(.05)
    desc=((full-mean)/std).clip(-5,5).astype(np.float32)
    # Cached 12,288 fingerprint dimensions, counts compressed monotonically.
    fp=d['views']['multiview'][:,:-435]
    assert np.min(fp)>=0
    fp=np.sqrt(fp).astype(np.float32)
    target=np.zeros(len(d['y']),np.float32);offset=np.zeros_like(target);qoffset=np.zeros_like(target);states=[];members=[]
    original=b.table(r88.r84.ORIGINAL)
    for k,metric in enumerate(d['metrics']):
        tr=fit[d['tid'][fit]==k];qq=q[d['tid'][q]==k];assert len(tr)>20
        base=r83.raw_proxy(d,tr);qp=(np.array([float(r['y_pred']) for r in original if r['task']==d['tasks'][k]]) if part=='final' else r83.raw_proxy(d,qq))
        y=d['y'][tr]
        if metric in ('AUROC','AUPRC'):
            z,qz=r83.ranklog(base),r83.ranklog(qp);beta=r83.calibrate(y,z)
            offset[tr]=beta[0]+beta[1]*z;qoffset[qq]=beta[0]+beta[1]*qz;target[tr]=y
            state=dict(metric=metric,beta=beta.tolist(),scale=1.,center=0.)
        elif metric=='MAE':
            scale=max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-3)
            target[tr]=(y-base)/scale;qoffset[qq]=qp
            state=dict(metric=metric,scale=scale,center=float(np.median(y)))
        else:
            z,qz=r83.ranknorm(base),r83.ranknorm(qp);xx=np.c_[np.ones(len(z)),z]
            beta=np.linalg.solve(xx.T@xx+np.diag([1e-8,1.]),xx.T@r83.ranknorm(y));beta[1]=max(beta[1],.01)
            target[tr]=r83.ranknorm(y)-xx@beta;qoffset[qq]=beta[0]+beta[1]*qz
            state=dict(metric=metric,beta=beta.tolist(),scale=1.,center=0.)
        states.append(state);members.append(tr)
    return dict(fit=fit,query=q,fp=fp,desc=desc,median=med,mean=mean,std=std,
        target=target,offset=offset,qoffset=qoffset,states=states,members=members)


class Net(nn.Module):
    def __init__(self,fp_dim,desc_dim,tasks):
        super().__init__()
        self.fp=nn.Sequential(nn.Linear(fp_dim,192),nn.LayerNorm(192),nn.SiLU())
        self.desc=nn.Sequential(nn.Linear(desc_dim,64),nn.LayerNorm(64),nn.SiLU())
        self.trunk=nn.Sequential(nn.Linear(256,192),nn.LayerNorm(192),nn.SiLU(),nn.Dropout(.3))
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(192,96),nn.SiLU(),nn.Dropout(.25)) for _ in range(3)])
        self.molecule_gate=nn.Linear(192,3);self.task_gate=nn.Parameter(torch.zeros(tasks,3))
        self.out=nn.Linear(96,tasks);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,fp,desc):
        h=self.trunk(torch.cat([self.fp(nn.functional.dropout(fp,.08,self.training)),self.desc(desc)],1))
        e=torch.stack([layer(h) for layer in self.experts],1)
        gate=(self.molecule_gate(h)[:,None]+self.task_gate[None]).softmax(-1)
        features=torch.einsum('bke,beh->bkh',gate,e)
        return (features*self.out.weight[None]).sum(-1)+self.out.bias


def infer(net,d,a,fp,desc):
    net.eval();q=a['query'];mols=np.unique(d['mi'][q]);values=np.empty((len(mols),len(d['tasks'])),np.float32)
    with torch.inference_mode():
        for start in range(0,len(mols),512):
            mm=mols[start:start+512];values[start:start+len(mm)]=net(fp[mm],desc[mm]).cpu().numpy()
    pos=np.searchsorted(mols,d['mi'][q]);return values[pos,d['tid'][q]].astype(float)


def train(d,part,seed):
    folder=OUT/'models'/part/str(seed);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==signature();np.testing.assert_array_equal(z['query'],split(d,part)[1]);return z['deltas'].copy()
    folder.mkdir(exist_ok=True,parents=True);started=time.monotonic();a=prepare(d,part)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    fp=torch.from_numpy(a['fp']).cuda();desc=torch.from_numpy(a['desc']).cuda()
    net=Net(fp.shape[1],desc.shape[1],len(d['tasks'])).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.02)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00005)
    target=torch.from_numpy(a['target']).cuda();offset=torch.from_numpy(a['offset']).cuda();snapshots=[];hashes={}
    for step in range(1,STEPS+1):
        net.train();k=(step-1)%len(d['tasks']);tr=a['members'][k];ix=rng.choice(tr,BATCH,replace=len(tr)<BATCH)
        mm=d['mi'][ix];delta=net(fp[mm],desc[mm])[:,k];metric=d['metrics'][k]
        if metric in ('AUROC','AUPRC'):loss=nn.functional.binary_cross_entropy_with_logits(offset[ix]+delta,target[ix])
        elif metric=='MAE':loss=.95*(delta-target[ix]).abs().mean()+.05*nn.functional.smooth_l1_loss(delta,target[ix],beta=.1)
        else:loss=nn.functional.mse_loss(delta,target[ix])
        loss=loss+.02*delta.square().mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();sched.step()
        if step%600==0:emit(stage='train',part=part,seed=seed,step=step,seconds=time.monotonic()-started,loss=float(loss.detach()))
        if step in SNAPSHOTS:
            delta=infer(net,d,a,fp,desc);snapshots.append(delta)
            p=folder/f'checkpoint_{step}.pt'
            torch.save(dict(signature=signature(),state={k:value.detach().cpu().clone() for k,value in net.state_dict().items()},
                fp_dim=fp.shape[1],desc_dim=desc.shape[1],tasks=len(d['tasks']),fit=a['fit'],query=a['query'],
                seed=seed,step=step,median=a['median'],mean=a['mean'],std=a['std'],states=a['states'],qoffset=a['qoffset'][a['query']]),p)
            hashes[p.name]=b.sha(p)
    deltas=np.stack(snapshots);np.savez_compressed(file,signature=signature(),fit=a['fit'],query=a['query'],deltas=deltas,snapshots=SNAPSHOTS,qoffset=a['qoffset'][a['query']])
    b.save(folder/'metadata.json',dict(signature=signature(),part=part,seed=seed,steps=STEPS,snapshots=SNAPSHOTS,
        checkpoint_hashes=hashes,predictions_sha256=b.sha(file),seconds=time.monotonic()-started,all_query_scaffolds_excluded=True))
    del net,opt,fp,desc,target,offset;torch.cuda.empty_cache();return deltas


def expert(d,k,part,checkpoint):
    ps=np.stack([train(d,part,s) for s in SEEDS]);j=SNAPSHOTS.index(checkpoint)
    f=OUT/'models'/part/str(SEEDS[0]);cache=np.load(f/'predictions.npz');q=cache['query'];active=d['tid'][q]==k
    ck=torch.load(f/f'checkpoint_{checkpoint}.pt',map_location='cpu',weights_only=False);state=ck['states'][k]
    raw=cache['qoffset'][active]+ps[:,j,active].mean(0)*state['scale']
    p=expit(raw) if d['metrics'][k] in ('AUROC','AUPRC') else raw
    return p.clip(0,100) if d['tasks'][k]=='ppbr_az' else p


def mix(metric,base,new,w):return r88.mix(metric,base,new,w)


def develop(d):
    OUT.mkdir(exist_ok=True)
    b.save(OUT/'protocol.json',dict(version=89,signature=signature(),baseline='v88',highest_official_baseline='v86',highest_official_score_user_reported=50.900,
        architecture='Shared structure residual network with molecule/task gates; zero-initialized correction output',steps=STEPS,snapshots=SNAPSHOTS,seeds=SEEDS,batch=BATCH,weights=WEIGHTS,
        selection_parts=['fold0','fold1'],confirmation_parts=['fold2','valid'],
        selection='Per task maximize minimum gain over fixed checkpoints/weights; >.003 relative MAE, .001 Spearman, .0005 AUROC/AP.',
        confirmation='All four gains positive; same two-seed delta ensemble, no confirmation fallback tuning.',
        query_scaffolds_removed_from_all_tasks=True,no_assay_input=True,test_labels_used=False,external_data=False,new_pretrained_weights=False,
        limitations='Historical proxies are not exact v86/v88 OOF; shared model fit excludes global scaffolds, baseline legacy models exclude only own-task scaffolds. Labels historically reused.'))
    for part in ('fold0','fold1'):
        for seed in SEEDS:train(d,part,seed)
    records=[]
    for k,task in enumerate(d['tasks']):
        metric=d['metrics'][k]
        for step in SNAPSHOTS:
            pp={part:expert(d,k,part,step) for part in ('fold0','fold1')}
            for w in WEIGHTS:
                parts=[]
                for part,p in pp.items():
                    q=r83.split(d,k,part)[1];base=baseline(d,k,part);pred=mix(metric,base,p,w)
                    parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],base),candidate=r83.score(metric,d['y'][q],pred),gain=r83.gain(metric,d['y'][q],base,pred)))
                records.append(dict(task=task,metric=metric,checkpoint=step,weight=w,parts=parts,objective=min(p['gain'] for p in parts)))
    selected=[]
    for task in d['tasks']:
        best=max((r for r in records if r['task']==task),key=lambda r:r['objective'])
        threshold=.003 if best['metric']=='MAE' else .001 if best['metric']=='Spearman' else .0005
        if best['objective']>threshold:selected.append(best)
        emit(stage='development',**best)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=signature(),records=records))
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=signature(),selected=selected,confirmation_inspected=False))


def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==signature();records=[]
    if frozen['selected']:
        for part in ('fold2','valid'):
            for seed in SEEDS:train(d,part,seed)
    for chosen in frozen['selected']:
        k=d['tasks'].index(chosen['task']);metric=chosen['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];base=baseline(d,k,part);p=mix(metric,base,expert(d,k,part,chosen['checkpoint']),chosen['weight'])
            parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],base),candidate=r83.score(metric,d['y'][q],p),gain=r83.gain(metric,d['y'][q],base,p)))
        r=dict(**chosen,confirmation=parts,accepted=all(p['gain']>0 for p in parts));records.append(r);emit(stage='confirmation',**r)
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=signature(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','confirm']);args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    d=load();(develop if args.stage=='develop' else confirm)(d)
