"""Balanced multitask direct/residual experts over the immutable v92 anchor.

Every update contains examples from all 22 tasks. Official-valid models use
all official train rows; global scaffold exclusions apply to CV only.
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
import v92_full_kernel as prev
b,r83=prev.b,prev.r83
OUT=ROOT/'admet_model_outputs_v93'
ANCHOR=ROOT/'admet_submissions/v92_full_kernel/我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
ANCHOR_SHA='5d740f5ba85ede9dcee4efdfb95556cee8cad7ba5075730045eb8146bcbbb2fa'
SEEDS=(934026,934027)
STEPS=6000
SNAPSHOTS=(1000,3000,6000)
PER_TASK=32
WEIGHTS=(.25,.5,1.)
MODES=('direct','residual')

def sig():return hashlib.sha256((b.sha(__file__)+prev.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def load():
    d=prev.load();assert b.sha(ANCHOR)==ANCHOR_SHA;return d

def base(d,k,part):
    if part=='final':return np.array([float(r['y_pred']) for r in b.table(ANCHOR) if r['task']==d['tasks'][k]])
    old=prev.base(d,k,part)
    c=json.loads((prev.OUT/'confirmation.json').read_text())
    chosen=next((r for r in c['records'] if r['task']==d['tasks'][k] and r['accepted']),None)
    if chosen:old=prev.mix(chosen['metric'],old,prev.expert(d,k,part,chosen['config']),chosen['weight'])
    return old

def split(d,part):
    if part=='final':fit=np.flatnonzero(d['split']<2);q=np.flatnonzero(d['split']==2)
    elif part=='valid':fit=np.flatnonzero(d['split']==0);q=np.flatnonzero(d['split']==1)
    else:
        q=np.flatnonzero((d['split']==0)&(d['legacy_folds']==int(part[-1])))
        fit=np.flatnonzero((d['split']==0)&~np.isin(d['groups'],d['groups'][q]))
        assert not set(d['groups'][fit])&set(d['groups'][q])
        assert not set(d['canonical'][fit])&set(d['canonical'][q])
    assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    for k in range(len(d['tasks'])):
        tr,qq=r83.split(d,k,part);np.testing.assert_array_equal(q[d['tid'][q]==k],qq)
        np.testing.assert_array_equal(fit[d['tid'][fit]==k],tr)
    return fit,q

def prepare(d,part):
    fit,q=split(d,part);mols=np.unique(d['mi'][fit]);raw=d['views']['physchem'].astype(float)
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in raw[mols].T])
    full=np.where(np.isfinite(raw),raw,med);mean=full[mols].mean(0);std=full[mols].std(0).clip(.05)
    desc=((full-mean)/std).clip(-5,5).astype(np.float32)
    fp=np.sqrt(d['views']['multiview'][:,:12288]).astype(np.float32)
    target=np.zeros((len(d['y']),2),np.float32);offset=np.zeros(len(d['y']),np.float32)
    qoff=np.zeros(len(q),np.float32);states=[];members=[]
    original=b.table(prev.prev.prev.r88.r84.ORIGINAL)
    for k,metric in enumerate(d['metrics']):
        tr=fit[d['tid'][fit]==k];active=d['tid'][q]==k;qq=q[active];y=d['y'][tr]
        bp=r83.raw_proxy(d,tr)
        qp=np.array([float(r['y_pred']) for r in original if r['task']==d['tasks'][k]]) if part=='final' else r83.raw_proxy(d,qq)
        if metric in ('AUROC','AUPRC'):
            beta=r83.calibrate(y,r83.ranklog(bp));offset[tr]=beta[0]+beta[1]*r83.ranklog(bp);qoff[active]=beta[0]+beta[1]*r83.ranklog(qp)
            target[tr,0]=target[tr,1]=y;state=dict(metric=metric,center=0.,scale=1.,beta=beta.tolist())
        elif metric=='MAE':
            center=float(np.median(y));scale=max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-3)
            target[tr,0]=(y-center)/scale;target[tr,1]=(y-bp)/scale;qoff[active]=qp
            state=dict(metric=metric,center=center,scale=scale)
        else:
            z=r83.ranknorm(bp);x=np.c_[np.ones(len(z)),z];beta=np.linalg.solve(x.T@x+np.diag([1e-8,1.]),x.T@r83.ranknorm(y));beta[1]=max(beta[1],.01)
            target[tr,0]=r83.ranknorm(y);target[tr,1]=target[tr,0]-x@beta;qoff[active]=beta[0]+beta[1]*r83.ranknorm(qp)
            state=dict(metric=metric,center=0.,scale=1.,beta=beta.tolist())
        states.append(state);members.append(tr)
    return dict(fit=fit,query=q,fp=fp,desc=desc,median=med,mean=mean,std=std,target=target,offset=offset,qoffset=qoff,states=states,members=members)

class Net(nn.Module):
    def __init__(self,desc_dim,tasks):
        super().__init__();self.desc_dim=desc_dim;self.tasks=tasks
        self.views=nn.ModuleList([nn.Sequential(nn.Linear(2048,64),nn.LayerNorm(64),nn.SiLU()) for _ in range(6)])
        self.desc=nn.Sequential(nn.Linear(desc_dim,64),nn.LayerNorm(64),nn.SiLU())
        self.trunk=nn.Sequential(nn.Linear(448,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(.35),nn.Linear(256,192),nn.LayerNorm(192),nn.SiLU(),nn.Dropout(.25))
        self.head_weight=nn.Parameter(torch.empty(tasks,192,48));nn.init.xavier_uniform_(self.head_weight)
        self.head_bias=nn.Parameter(torch.zeros(tasks,48));self.out_weight=nn.Parameter(torch.empty(tasks,48,2));nn.init.normal_(self.out_weight,std=.005)
        self.out_bias=nn.Parameter(torch.zeros(tasks,2))
    def forward(self,fp,desc,tid):
        x=[layer(nn.functional.dropout(fp[:,i*2048:(i+1)*2048],.08,self.training)) for i,layer in enumerate(self.views)]
        h=self.trunk(torch.cat(x+[self.desc(desc)],1))
        h=nn.functional.silu(torch.bmm(h[:,None],self.head_weight[tid]).squeeze(1)+self.head_bias[tid])
        h=nn.functional.dropout(h,.25,self.training)
        return torch.bmm(h[:,None],self.out_weight[tid]).squeeze(1)+self.out_bias[tid]

def infer(net,d,a,fp,desc,batch=512):
    net.eval();q=a['query'];out=[]
    with torch.inference_mode():
        for start in range(0,len(q),batch):
            ix=q[start:start+batch];mm=d['mi'][ix];tid=torch.as_tensor(d['tid'][ix],device=fp.device)
            out.append(net(fp[mm],desc[mm],tid).cpu().numpy())
    return np.concatenate(out).astype(float)

def train(d,part,seed):
    folder=OUT/'models'/part/str(seed);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);meta=json.loads((folder/'metadata.json').read_text());assert str(z['signature'])==meta['signature']==sig() and b.sha(file)==meta['predictions_sha256']
        np.testing.assert_array_equal(z['query'],split(d,part)[1]);return z['values'].copy()
    folder.mkdir(parents=True,exist_ok=True);started=time.monotonic();a=prepare(d,part)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    fp=torch.from_numpy(a['fp']).cuda();desc=torch.from_numpy(a['desc']).cuda();target=torch.from_numpy(a['target']).cuda();offset=torch.from_numpy(a['offset']).cuda()
    net=Net(desc.shape[1],len(d['tasks'])).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.00035,weight_decay=.025)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00003)
    kinds=torch.as_tensor([0 if m in ('AUROC','AUPRC') else 1 if m=='MAE' else 2 for m in d['metrics']],device='cuda')
    snapshots=[];hashes={};losses=[]
    for step in range(1,STEPS+1):
        net.train();ix=np.concatenate([rng.choice(tr,PER_TASK,replace=len(tr)<PER_TASK) for tr in a['members']]);tid=torch.as_tensor(d['tid'][ix],device='cuda');mm=d['mi'][ix]
        out=net(fp[mm],desc[mm],tid);yt=target[ix];kind=kinds[tid];off=offset[ix]
        classification=nn.functional.binary_cross_entropy_with_logits(out+torch.stack([torch.zeros_like(off),off],1),yt.clamp(0,1),reduction='none')
        mae=(out-yt).abs();mse=(out-yt).square()
        loss=torch.where((kind==0)[:,None],classification,torch.where((kind==1)[:,None],mae,mse)).mean()+.005*out.square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();sched.step()
        if step%500==0:
            losses.append(dict(step=step,loss=float(loss.detach())));emit(stage='train',part=part,seed=seed,step=step,updates_per_task=step,seconds=time.monotonic()-started,loss=float(loss.detach()))
        if step in SNAPSHOTS:
            snapshots.append(infer(net,d,a,fp,desc));p=folder/f'checkpoint_{step}.pt'
            torch.save(dict(signature=sig(),state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},desc_dim=desc.shape[1],tasks=len(d['tasks']),fit=a['fit'],query=a['query'],seed=seed,step=step,median=a['median'],mean=a['mean'],std=a['std'],states=a['states'],qoffset=a['qoffset']),p);hashes[p.name]=b.sha(p)
    values=np.stack(snapshots);np.savez_compressed(file,signature=sig(),fit=a['fit'],query=a['query'],values=values,snapshots=SNAPSHOTS,qoffset=a['qoffset'])
    b.save(folder/'metadata.json',dict(signature=sig(),part=part,seed=seed,steps=STEPS,snapshots=SNAPSHOTS,checkpoint_hashes=hashes,predictions_sha256=b.sha(file),seconds=time.monotonic()-started,losses=losses))
    del net,opt,fp,desc,target,offset;torch.cuda.empty_cache();return values

def expert(d,k,part,cfg):
    memo=d.setdefault('_v93_experts',{})
    if part not in memo:
        values=np.mean([train(d,part,s) for s in SEEDS],axis=0)
        f=OUT/'models'/part/str(SEEDS[0]);cache=dict(np.load(f/'predictions.npz'))
        ck=torch.load(f/f'checkpoint_{SNAPSHOTS[0]}.pt',map_location='cpu',weights_only=False)
        memo[part]=(values,cache['query'],cache['qoffset'],ck['states'])
    values,query,qoff,states=memo[part];j=SNAPSHOTS.index(cfg['step']);mode=MODES.index(cfg['mode']);active=d['tid'][query]==k;state=states[k]
    raw=values[j,active,mode]*state['scale']+(state['center'] if mode==0 else qoff[active])
    p=expit(raw) if d['metrics'][k] in ('AUROC','AUPRC') else raw
    return p.clip(0,100) if d['tasks'][k]=='ppbr_az' else p

def threshold(metric):return .01 if metric=='MAE' else .005 if metric=='Spearman' else .002
def mix(metric,old,p,w):return prev.mix(metric,old,p,w)
def develop(d):
    OUT.mkdir(exist_ok=True)
    b.save(OUT/'protocol.json',dict(version=93,signature=sig(),baseline='v92',anchor_sha256=ANCHOR_SHA,baseline_official_score_user_reported=50.915,seeds=SEEDS,steps=STEPS,snapshots=SNAPSHOTS,modes=MODES,weights=WEIGHTS,batch_per_task=PER_TASK,updates_per_task=STEPS,selection_parts=['fold0','fold1'],confirmation_parts=['fold2','valid'],selection='Max minimum gain >1% relative MAE, .005 Spearman, .002 AUROC/AP; one config per task frozen.',confirmation='All four positive; mean gain over threshold; no fallback tuning.',training='All task-balanced batches; paired direct and residual heads. CV global scaffolds excluded; valid all official train; final train+valid.',test_labels_used=False,external_data=False,new_pretrained_weights=False,limitations='Historical proxies not exact v92 OOF. Labels reused historically. Other-task train measurements may concern valid molecules; this is official-only multitask learning. No official points estimate.'))
    b.save(OUT/'data_audit.json',dict(input_hashes=d['sources'],test_labels_finite=0,legacy_folds_sha256=b.sha(ROOT/'recovered_history/legacy_v24_folds.npy')))
    for part in ('fold0','fold1'):
        for seed in SEEDS:train(d,part,seed)
    records=[];selected=[]
    for k,t in enumerate(d['tasks']):
        metric=d['metrics'][k];bases={part:base(d,k,part) for part in ('fold0','fold1')}
        for step in SNAPSHOTS:
            for mode in MODES:
                cfg=dict(step=step,mode=mode);ps={part:expert(d,k,part,cfg) for part in bases}
                for w in WEIGHTS:
                    parts=[]
                    for part,old in bases.items():
                        q=r83.split(d,k,part)[1];p=mix(metric,old,ps[part],w)
                        parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],old),candidate=r83.score(metric,d['y'][q],p),gain=r83.gain(metric,d['y'][q],old,p)))
                    records.append(dict(task=t,metric=metric,config=cfg,weight=w,parts=parts,objective=min(x['gain'] for x in parts)))
        best=max((r for r in records if r['task']==t),key=lambda r:r['objective']);emit(stage='development',**best)
        if best['objective']>threshold(metric):selected.append(best)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records));b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected,confirmation_inspected=False));emit(stage='frozen',selected=selected)

def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==sig();records=[]
    if frozen['selected']:
        for part in ('fold2','valid'):
            for seed in SEEDS:train(d,part,seed)
    for chosen in frozen['selected']:
        k=d['tasks'].index(chosen['task']);metric=chosen['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];old=base(d,k,part);p=mix(metric,old,expert(d,k,part,chosen['config']),chosen['weight'])
            parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],old),candidate=r83.score(metric,d['y'][q],p),gain=r83.gain(metric,d['y'][q],old,p)))
        record=dict(**chosen,confirmation=parts,accepted=all(x['gain']>0 for x in parts) and np.mean([x['gain'] for x in parts])>threshold(metric));records.append(record);emit(stage='confirmation',**record)
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))

def smoke(d):
    OUT.mkdir(exist_ok=True);a=prepare(d,'fold0');net=Net(a['desc'].shape[1],len(d['tasks'])).cuda();q=a['query'][:37]
    fp=torch.from_numpy(a['fp']).cuda();desc=torch.from_numpy(a['desc']).cuda();aa=dict(a,query=q)
    x=infer(net,d,aa,fp,desc);y=infer(net,d,aa,fp,desc,11);np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-5)
    # Analytic identity for the mask-reduced balanced loss, with negative regression targets.
    p=torch.tensor([[.2,.3],[-2.,1.],[1.,-1.]],device='cuda',requires_grad=True);yt=torch.tensor([[0.,1.],[-3.,-2.],[.2,.4]],device='cuda');kind=torch.tensor([0,1,2],device='cuda')
    cls=nn.functional.binary_cross_entropy_with_logits(p,yt.clamp(0,1),reduction='none');loss=torch.where((kind==0)[:,None],cls,torch.where((kind==1)[:,None],(p-yt).abs(),(p-yt).square())).mean()
    expected=(nn.functional.binary_cross_entropy_with_logits(p[:1],yt[:1])+(p[1]-yt[1]).abs().mean()+(p[2]-yt[2]).square().mean())/3;torch.testing.assert_close(loss,expected);loss.backward();assert torch.isfinite(p.grad).all()
    p=OUT/'smoke.pt';torch.save(net.state_dict(),p);other=Net(a['desc'].shape[1],len(d['tasks'])).cuda();other.load_state_dict(torch.load(p,weights_only=True));np.testing.assert_allclose(x,infer(other,d,aa,fp,desc),atol=0,rtol=0)
    for part in ('fold0','fold1','fold2','valid','final'):split(d,part)
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),alternate_batch=True,model_reload=True,balanced_loss_and_gradient=True,all_split_membership=True,test_labels_finite=0));emit(stage='smoke',status='PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','develop','confirm']);args=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](load())
