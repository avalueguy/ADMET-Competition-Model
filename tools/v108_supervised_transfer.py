"""Official-only supervised representation learning and endpoint adaptation.

One fixed architecture/objective/blend, with a matched scratch control. This
does not claim that historical v101 proxy arrays are a fully refitted baseline.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit, ndtri
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import admet_v77_ranking as data
import v83_residual as r83
OUT = ROOT/'admet_model_outputs_v108'
SEEDS = (1084026, 1084027)
PRE_STEPS, FINE_STEPS, BATCH = 2200, 1000, 192
WEIGHT = .35
ANCHOR = ROOT/'admet_submissions/v101_graph_assay_shape/我从山北来带着兰花草-小分子药物预测技术创新赛-20261026.csv'
ANCHOR_SHA = 'e94961c81deedf98bcb69e04d621a54ae9319aab47bb1ced9d97890f13c8f086'

def sig():
    return hashlib.sha256((data.sha(__file__)+data.signature()+data.sha(ROOT/'recovered_history/legacy_v24_folds.npy')+ANCHOR_SHA).encode()).hexdigest()

def emit(**x):
    print(json.dumps(x, ensure_ascii=False), flush=True)

def load():
    d = data.load_data()
    d['legacy_folds'] = np.load(ROOT/'recovered_history/legacy_v24_folds.npy')
    assert data.sha(ANCHOR) == ANCHOR_SHA
    return d

def split(d, part):
    if part == 'final':
        fit, q = np.flatnonzero(d['split'] < 2), np.flatnonzero(d['split'] == 2)
    elif part == 'valid':
        fit, q = np.flatnonzero(d['split'] == 0), np.flatnonzero(d['split'] == 1)
    elif part in ('random', 'scaffold'):
        field = 'canonical' if part == 'random' else 'groups'
        # Deterministic new partitions. These are reused public labels, not new data.
        hold = np.array([int(hashlib.sha256(('108-confirm-'+str(x)).encode()).hexdigest()[:12],16)%5==0 for x in d[field]])
        fit = np.flatnonzero((d['split'] == 0) & ~hold)
        q = np.flatnonzero((d['split'] == 0) & hold)
        assert not set(d[field][fit]) & set(d[field][q])
    else:
        q = np.flatnonzero((d['split']==0)&(d['legacy_folds']==int(part[-1])))
        fit = np.flatnonzero((d['split']==0)&~np.isin(d['groups'],d['groups'][q]))
        assert not set(d['groups'][fit]) & set(d['groups'][q])
        for k in range(len(d['tasks'])):
            tr, qq = r83.split(d,k,part)
            np.testing.assert_array_equal(fit[d['tid'][fit]==k],tr)
            np.testing.assert_array_equal(q[d['tid'][q]==k],qq)
    assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    return fit,q

def design(d, part):
    fit, query = split(d,part)
    mm = np.unique(d['mi'][fit]); raw = d['views']['physchem'].astype(float)
    med = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in raw[mm].T])
    full = np.where(np.isfinite(raw),raw,med)
    mean, std = full[mm].mean(0), full[mm].std(0).clip(.05)
    desc = ((full-mean)/std).clip(-6,6).astype(np.float32)
    fp = np.sqrt(d['views']['multiview'][:,:12288]).astype(np.float32)
    target = np.zeros(len(d['y']),np.float32); states=[]; members=[]
    for k,m in enumerate(d['metrics']):
        tr = fit[d['tid'][fit]==k]; yy = d['y'][tr]
        if m in ('AUROC','AUPRC'):
            center,scale=0.,1.; z=yy
        elif m=='MAE':
            center=float(np.median(yy));scale=max(float(np.subtract(*np.quantile(yy,[.75,.25]))),.001)
            z=(yy-center)/scale
        else:
            center,scale=0.,1.;z=ndtri(np.clip((rankdata(yy)-.5)/len(yy),1e-5,1-1e-5))
        target[tr]=z;members.append(tr)
        states.append(dict(metric=m,center=center,scale=scale))
    return dict(fit=fit,query=query,fp=fp,desc=desc,median=med,mean=mean,std=std,
                target=target,states=states,members=members)

class Net(nn.Module):
    def __init__(self, desc_dim=435, tasks=22):
        super().__init__()
        def block(a,b,p):
            return nn.Sequential(nn.Linear(a,b),nn.LayerNorm(b),nn.SiLU(),nn.Dropout(p))
        self.views=nn.ModuleList(block(2048,96,.15) for _ in range(6))
        self.desc=block(desc_dim,96,.1)
        self.trunk=nn.Sequential(block(672,384,.3),block(384,192,.2))
        self.heads=nn.ModuleList(nn.Sequential(block(192,64,.2),nn.Linear(64,1)) for _ in range(tasks))
    def encode(self,fp,desc):
        return self.trunk(torch.cat([layer(nn.functional.dropout(fp[:,i*2048:(i+1)*2048],.05,self.training)) for i,layer in enumerate(self.views)]+[self.desc(desc)],1))
    def forward(self,fp,desc,k=None):
        h=self.encode(fp,desc)
        return torch.cat([head(h) for head in self.heads],1) if k is None else self.heads[k](h).squeeze(1)

def loss_vector(p,y,kind):
    cls=nn.functional.binary_cross_entropy_with_logits(p,y.clamp(0,1),reduction='none')
    return torch.where(kind==0,cls,torch.where(kind==1,(p-y).abs(),(p-y).square()))

def task_loss(p,y,metric):
    if metric in ('AUROC','AUPRC'):
        loss=nn.functional.binary_cross_entropy_with_logits(p,y)
        # A fixed ranking auxiliary improves discrimination without selecting a
        # different hyperparameter for each endpoint.
        if metric=='AUPRC':
            positive=p[y>.5];negative=p[y<.5]
            if len(positive) and len(negative):
                loss=loss+.15*nn.functional.softplus(negative[:,None]-positive[None]).mean()
    elif metric=='MAE':
        loss=(p-y).abs().mean()
    else:
        delta=y[:,None]-y[None];mask=delta.abs()>.15
        pair=nn.functional.softplus(-delta.sign()*(p[:,None]-p[None]))
        loss=.5*(p-y).square().mean()+.5*pair[mask].mean()
    return loss+.001*p.square().mean()

def infer(net,d,a,fp,desc,k=None,batch=512,device='cuda'):
    q=a['query'] if k is None else a['query'][d['tid'][a['query']]==k]
    result=[];net.eval()
    with torch.inference_mode():
        for lo in range(0,len(q),batch):
            ix=q[lo:lo+batch];mm=d['mi'][ix]
            p=net(fp[mm],desc[mm],k)
            if k is None:p=p[torch.arange(len(ix),device=device),torch.as_tensor(d['tid'][ix],device=device)]
            result.append(p.cpu().numpy())
    return np.concatenate(result).astype(float)

def state_copy(net):
    return {n:p.detach().clone() for n,p in net.state_dict().items()}

def average_state(average,net,count):
    for n,p in net.state_dict().items():
        average[n].add_((p.detach()-average[n])/count)

def cached(folder):
    p=folder/'predictions.npz'
    if not p.exists():return None
    z=dict(np.load(p));meta=json.loads((folder/'metadata.json').read_text())
    assert str(z['signature'])==meta['signature']==sig()
    assert data.sha(p)==meta['prediction_sha256'] and data.sha(folder/'model.pt')==meta['model_sha256']
    return z

def save_model(folder,net,a,fit,query,raw,seed,part,kind,seconds,k=None):
    folder.mkdir(parents=True,exist_ok=True)
    model=dict(signature=sig(),weights={n:p.detach().cpu() for n,p in net.state_dict().items()},
               desc_dim=a['desc'].shape[1],tasks=len(a['states']),fit=fit,query=query,states=a['states'],
               median=a['median'],mean=a['mean'],std=a['std'],seed=seed,part=part,kind=kind,k=k)
    torch.save(model,folder/'model.pt')
    np.savez_compressed(folder/'predictions.npz',signature=sig(),fit=fit,query=query,raw=raw)
    data.save(folder/'metadata.json',dict(signature=sig(),model_sha256=data.sha(folder/'model.pt'),
        prediction_sha256=data.sha(folder/'predictions.npz'),seconds=seconds,seed=seed,part=part,kind=kind,k=k))

def pretrain(d,part,seed,a=None):
    folder=OUT/'pretrain'/part/str(seed);z=cached(folder)
    if z is not None:return folder
    started=time.monotonic();a=design(d,part) if a is None else a
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    fp=torch.as_tensor(a['fp'],device='cuda');desc=torch.as_tensor(a['desc'],device='cuda')
    target=torch.as_tensor(a['target'],device='cuda');net=Net(desc.shape[1],len(d['tasks'])).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.02)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,PRE_STEPS,eta_min=.00008)
    kinds=torch.as_tensor([0 if m in ('AUROC','AUPRC') else 1 if m=='MAE' else 2 for m in d['metrics']],device='cuda')
    probs=np.sqrt([len(m) for m in a['members']]);probs=probs/probs.sum()
    avg=None;count=0
    for step in range(1,PRE_STEPS+1):
        net.train();ts=rng.choice(len(d['tasks']),384,p=probs)
        ix=np.array([rng.choice(a['members'][k]) for k in ts]);mm=d['mi'][ix];tid=torch.as_tensor(ts,device='cuda')
        p=net(fp[mm],desc[mm])[torch.arange(len(ix),device='cuda'),tid]
        loss=loss_vector(p,target[ix],kinds[tid]).mean()+.001*p.square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step();sched.step()
        if step>PRE_STEPS*.8 and step%20==0:
            count+=1
            if avg is None:avg=state_copy(net)
            else:average_state(avg,net,count)
        if step%550==0:emit(stage='pretrain',part=part,seed=seed,step=step,loss=float(loss.detach()),seconds=round(time.monotonic()-started))
    net.load_state_dict(avg);raw=infer(net,d,a,fp,desc)
    save_model(folder,net,a,a['fit'],a['query'],raw,seed,part,'pretrain',time.monotonic()-started)
    del fp,desc,net,opt,target;torch.cuda.empty_cache();return folder

def train_task(d,k,part,seed,kind,a=None):
    folder=OUT/'models'/d['tasks'][k]/part/kind/str(seed);z=cached(folder)
    if z is not None:return z['raw']
    a=design(d,part) if a is None else a;started=time.monotonic()
    torch.manual_seed(seed+k*101);torch.cuda.manual_seed_all(seed+k*101);rng=np.random.default_rng(seed+k*101)
    net=Net(a['desc'].shape[1],len(d['tasks'])).cuda()
    if kind=='transfer':
        path=pretrain(d,part,seed,a)
        ck=torch.load(path/'model.pt',map_location='cuda',weights_only=False);net.load_state_dict(ck['weights']);del ck
        # Retain source head initialization; task adaptation updates its own head
        # and the encoder, with a lower encoder learning rate.
    head=list(net.heads[k].parameters());shared=list(net.views.parameters())+list(net.desc.parameters())+list(net.trunk.parameters())
    opt=torch.optim.AdamW([dict(params=shared,lr=.00012 if kind=='transfer' else .0006),dict(params=head,lr=.0006)],weight_decay=.03)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,FINE_STEPS,eta_min=.000025)
    tr=a['members'][k];q=a['query'][d['tid'][a['query']]==k]
    fp=torch.as_tensor(a['fp'],device='cuda');desc=torch.as_tensor(a['desc'],device='cuda');target=torch.as_tensor(a['target'],device='cuda')
    avg=None;count=0
    for step in range(1,FINE_STEPS+1):
        net.train();ix=rng.choice(tr,BATCH,replace=len(tr)<BATCH);mm=d['mi'][ix]
        p=net(fp[mm],desc[mm],k);loss=task_loss(p,target[ix],d['metrics'][k])
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(head+shared,3.);opt.step();sched.step()
        if step>FINE_STEPS*.8 and step%20==0:
            count+=1
            if avg is None:avg=state_copy(net)
            else:average_state(avg,net,count)
    net.load_state_dict(avg);raw=infer(net,d,a,fp,desc,k)
    save_model(folder,net,a,tr,q,raw,seed,part,kind,time.monotonic()-started,k)
    emit(stage='adapt',task=d['tasks'][k],part=part,seed=seed,kind=kind,seconds=round(time.monotonic()-started))
    del net,opt,fp,desc,target;torch.cuda.empty_cache();return raw

def predictions(d,k,part,kind,a=None):
    raw=np.mean([train_task(d,k,part,s,kind,a) for s in SEEDS],axis=0)
    state=(design(d,part) if a is None else a)['states'][k]
    p=raw*state['scale']+state['center']
    if d['metrics'][k] in ('AUROC','AUPRC'):p=expit(p)
    return p.clip(0,100) if d['tasks'][k]=='ppbr_az' else p

def baseline(d,k,part):
    if '_baseline_module' not in d:
        import v102_count_margin as old
        loaded=old.load()
        for key in ('fingerprint','neutral_identity','conformer_features'):
            d[key]=loaded[key]
        del loaded
        release=json.loads((ROOT/'admet_model_outputs_v101/release.json').read_text())
        d['v101_selected']=release['selected'];d['_baseline_module']=old
    return d['_baseline_module'].base(d,k,part)

def mixed(metric,old,p):
    return (1-WEIGHT)*old+WEIGHT*p if metric=='MAE' else (1-WEIGHT)*rankdata(old)/len(old)+WEIGHT*rankdata(p)/len(p)

def threshold(metric):
    return .015 if metric=='MAE' else .012 if metric=='Spearman' else .005

def protocol():
    doc=dict(version=108,signature=sig(),anchor='v101',anchor_sha256=ANCHOR_SHA,official_best=50.932,
        seeds=SEEDS,pretrain_steps=PRE_STEPS,finetune_steps=FINE_STEPS,batch=BATCH,blend_weight=WEIGHT,
        architecture='Six view MLP encoders 96 each plus physicochemical96; trunk384/192; independent64 heads. Official multitask supervised training then task-specific full-network adaptation with lower encoder LR. SWA last20%.',
        control='Identical architecture, task objective, fine tuning length and SWA trained from scratch on identical fit rows. Historical v101 comparisons separately labeled approximate.',
        objective='BCE; AUPRC adds .15pair logistic; MAE L1; Spearman .5 normal-rank MSE + .5pair logistic.',
        candidates='One fixed transfer configuration and .35 blend for all tasks. No checkpoint/weight search.',
        development='Legacy fold0/1. Both blend gains positive, mean >1.5% relative MAE/.012rho/.005AUC or AP; transfer must beat scratch in both.',
        confirmation='Frozen endpoint list; fold2 and official valid gains positive; pooled 3fold+valid grouped bootstrap 95% lower bound>0; transfer vs scratch additionally nonnegative on newly partitioned canonical-random and scaffold holdouts. No failed fallback.',
        data='Official train Y only for all development; final adds official valid Y. Test structures only from immutable template. No measured assays in inputs. No external data or weights.',
        limitations='All public labels reused across past experiments; new partitions do not create independent data. v101 reconstructed caches/proxies are not full refits. No official score estimate.')
    p=OUT/'protocol.json';OUT.mkdir(exist_ok=True)
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(doc))
    else:data.save(p,doc)
    return doc

def run_part(d,part,tasks=None,kinds=('transfer','scratch')):
    a=design(d,part)
    for seed in SEEDS:
        if 'transfer' in kinds:pretrain(d,part,seed,a)
    for k in range(len(d['tasks'])):
        if tasks is not None and d['tasks'][k] not in tasks:continue
        for kind in kinds:
            predictions(d,k,part,kind,a)

def develop(d):
    protocol();records=[]
    for part in ('fold0','fold1'):run_part(d,part)
    for k,t in enumerate(d['tasks']):
        m=d['metrics'][k];parts=[]
        for part in ('fold0','fold1'):
            a=design(d,part);q=a['query'][d['tid'][a['query']]==k];y=d['y'][q]
            p=predictions(d,k,part,'transfer',a);s=predictions(d,k,part,'scratch',a);old=baseline(d,k,part);new=mixed(m,old,p)
            parts.append(dict(part=part,rows=len(q),baseline_proxy=r83.score(m,y,old),transfer=r83.score(m,y,p),scratch=r83.score(m,y,s),
                              candidate=r83.score(m,y,new),gain=r83.gain(m,y,old,new),transfer_gain=r83.gain(m,y,s,p)))
        selected=bool(all(x['gain']>0 and x['transfer_gain']>0 for x in parts) and np.mean([x['gain'] for x in parts])>threshold(m))
        record=dict(task=t,metric=m,parts=parts,selected=selected);records.append(record);emit(stage='development',**record)
    data.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records))
    data.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=[r for r in records if r['selected']],confirmation_inspected=False))

def bootstrap(d,k,rows,y,old,p):
    groups,inv=np.unique(d['groups'][rows],return_inverse=True);blocks=[np.flatnonzero(inv==j) for j in range(len(groups))]
    rng=np.random.default_rng(1080026+k);values=[]
    for _ in range(2000):
        ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
        if d['metrics'][k] in ('AUROC','AUPRC') and len(np.unique(y[ix]))<2:continue
        gain=r83.gain(d['metrics'][k],y[ix],old[ix],p[ix])
        if np.isfinite(gain):values.append(gain)
    return dict(percentile_95=np.quantile(values,[.025,.975]).tolist(),positive_fraction=float(np.mean(np.array(values)>0)),groups=len(groups),selection_adjusted=False)

def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==sig()
    tasks=[r['task'] for r in frozen['selected']]
    if tasks:
        for part in ('fold2','valid','random','scaffold'):run_part(d,part,tasks)
    records=[]
    for ch in frozen['selected']:
        k=d['tasks'].index(ch['task']);m=ch['metric'];parts=[];rows=[];yy=[];oo=[];pp=[]
        for part in ('fold0','fold1','fold2','valid','random','scaffold'):
            a=design(d,part);q=a['query'][d['tid'][a['query']]==k];y=d['y'][q]
            p=predictions(d,k,part,'transfer',a);s=predictions(d,k,part,'scratch',a)
            rec=dict(part=part,rows=len(q),transfer=r83.score(m,y,p),scratch=r83.score(m,y,s),transfer_gain=r83.gain(m,y,s,p))
            if part not in ('random','scaffold'):
                old=baseline(d,k,part);new=mixed(m,old,p)
                rec.update(baseline_proxy=r83.score(m,y,old),candidate=r83.score(m,y,new),gain=r83.gain(m,y,old,new))
                # Classification/ranking scores have separate scales across fits.
                op=rankdata(old)/len(old) if m!='MAE' else old
                npred=rankdata(new)/len(new) if m!='MAE' else new
                rows.extend(q);yy.extend(y);oo.extend(op);pp.extend(npred)
            parts.append(rec)
        ci=bootstrap(d,k,np.array(rows),np.array(yy),np.array(oo),np.array(pp))
        accepted=bool(all(x.get('gain',1)>0 for x in parts) and all(x['transfer_gain']>=0 for x in parts if x['part'] in ('random','scaffold')) and ci['percentile_95'][0]>0)
        r=dict(task=ch['task'],metric=m,confirmation=parts,uncertainty=ci,accepted=accepted);records.append(r);emit(stage='confirmation',**r)
    data.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=data.sha(OUT/'frozen_selection.json')))

def smoke(d):
    protocol();a=design(d,'fold0');q=a['query'];saved=d['y'][q].copy();d['y'][q]=1e6
    z=design(d,'fold0');d['y'][q]=saved
    for key in ('fit','query','fp','desc','target','median','mean','std'):np.testing.assert_array_equal(a[key],z[key])
    assert a['states']==z['states']
    k=d['tasks'].index('caco2_wang');torch.manual_seed(SEEDS[0]);net=Net(a['desc'].shape[1],len(d['tasks'])).cuda()
    fp=torch.as_tensor(a['fp'],device='cuda');desc=torch.as_tensor(a['desc'],device='cuda');tr=a['members'][k];ix=tr[:BATCH]
    opt=torch.optim.AdamW(net.parameters(),lr=.0005);started=time.monotonic()
    for _ in range(40):
        net.train();loss=task_loss(net(fp[d['mi'][ix]],desc[d['mi'][ix]],k),torch.as_tensor(a['target'][ix],device='cuda'),'MAE')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    seconds=time.monotonic()-started;p=infer(net,d,a,fp,desc,k);np.testing.assert_allclose(p,infer(net,d,a,fp,desc,k,73),atol=3e-6,rtol=3e-5)
    path=OUT/'smoke.pt';torch.save(net.state_dict(),path);cpu=Net(a['desc'].shape[1],len(d['tasks']));cpu.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    np.testing.assert_allclose(p,infer(cpu,d,a,fp.cpu(),desc.cpu(),k,device='cpu'),atol=2e-5,rtol=2e-5)
    data.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),query_label_perturbation=True,cpu_reload=True,batch_invariance=True,seconds_40_steps=seconds))
    emit(stage='smoke',status='PASS',seconds_40_steps=seconds)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['protocol','smoke','develop','confirm','part']);ap.add_argument('--part');args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    if args.stage=='protocol':protocol()
    else:
        d=load()
        if args.stage=='part':protocol();run_part(d,args.part)
        else:dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](d)
