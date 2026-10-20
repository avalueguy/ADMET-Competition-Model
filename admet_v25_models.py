"""V25 remote-only extended multitask training and independent task adaptation."""
import json,time,hashlib,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
import admet_v24_models as base

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_model_outputs_v25'
P=json.loads((ROOT/'admet_v25_protocol.json').read_text())

def signature():
    files=['admet_v25_protocol.json','admet_v25_models.py','admet_v25_run.py']
    return hashlib.sha256(json.dumps(dict(base_signature=base.signature(),files={p:base.sha(ROOT/p) for p in files}),sort_keys=True).encode()).hexdigest()

def train(d,fit,query,seed,dest,selected=None):
    assert os.name!='nt' and ROOT==Path(P['remote_root']) and torch.cuda.is_available()
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);cache=dest/'predictions.npz';sig=signature()
    if selected is None:selected=list(range(len(d['tasks'])))
    if cache.exists():
        z=np.load(cache);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],query)
        np.testing.assert_array_equal(z['selected'],selected)
        return z['p'].copy()
    assert not set(fit)&set(query) and np.all(d['split'][fit]<2)
    if np.any(d['split'][query]!=2):
        assert not set(d['rg'][fit])&set(d['rg'][query])
        assert not set(d['canonical'][d['mi'][fit]])&set(d['canonical'][d['mi'][query]])
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    n=len(d['tasks']);targets=[];train_ix=[];yy=[];weights=[];pw=[]
    for k in range(n):
        ix=fit[d['tid'][fit]==k];assert len(ix)
        target=base.Target(d['y'][ix],d['metrics'][k]);targets.append(target);train_ix.append(ix)
        yy.append(torch.tensor(target.encode(d['y'][ix]),device='cuda'))
        _,inv,counts=np.unique(d['canonical'][d['mi'][ix]],return_inverse=True,return_counts=True)
        weights.append(torch.tensor(1/np.sqrt(counts[inv]),dtype=torch.float32,device='cuda'))
        prevalence=float(d['y'][ix].mean()) if d['metrics'][k] in ['AUROC','AUPRC'] else 1.
        pw.append(min(4.,max(1.,np.sqrt((1-prevalence)/max(prevalence,.01)))))
    unique=np.unique(d['mi'][fit]);raw=np.concatenate([d['desc'],d['emb']],1)
    pre=base.Preprocessor().fit(raw[unique]);xs=[torch.tensor(x,device='cuda') for x in [d['fp2'],d['fcfp'],pre.transform(raw)]]
    dims=[x.shape[1] for x in xs];net=base.GatedNet(dims,n).cuda()
    counts=np.array([len(x) for x in train_ix]);prob=.5/n+.5*np.sqrt(counts)/np.sqrt(counts).sum()
    opt=torch.optim.AdamW(net.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,P['steps'],eta_min=P['minimum_lr'])
    updates=np.zeros(n,int);t0=time.time()

    def predict(k=None):
        net.eval();result=np.full(len(query),np.nan)
        with torch.inference_mode():
            for task in range(n) if k is None else [k]:
                jj=np.flatnonzero(d['tid'][query]==task)
                for start in range(0,len(jj),512):
                    active=jj[start:start+512];rr=d['mi'][query[active]]
                    logits=net([x[rr] for x in xs],task).cpu().numpy()
                    result[active]=targets[task].decode(logits)
        return result

    def step_task(k,optimizer):
        net.train();ix=train_ix[k];ii=rng.integers(len(ix),size=min(P['batch_size'],len(ix)))
        rr=d['mi'][ix[ii]];p=net([x[rr] for x in xs],k);by=yy[k][ii];m=d['metrics'][k];pair=0.
        if m in ['AUROC','AUPRC']:
            loss=nn.functional.binary_cross_entropy_with_logits(p,by,pos_weight=p.new_tensor(pw[k]),reduction='none')
            a=p[by>.5];b=p[by<.5]
            if len(a) and len(b):pair=.1*nn.functional.softplus(b[None]-a[:,None]).mean()
        elif m=='MAE':loss=.7*(p-by).abs()+.3*nn.functional.smooth_l1_loss(p,by,beta=.25,reduction='none')
        else:
            loss=nn.functional.smooth_l1_loss(p,by,beta=.3,reduction='none')
            diff=by[:,None]-by[None,:];mask=diff.abs()>.15
            if mask.any():pair=.15*nn.functional.softplus(-diff[mask].sign()*(p[:,None]-p[None,:])[mask]).mean()
        total=(loss*weights[k][ii]).sum()/weights[k][ii].sum()+pair
        assert torch.isfinite(total)
        optimizer.zero_grad(set_to_none=True);total.backward();nn.utils.clip_grad_norm_(net.parameters(),2.)
        optimizer.step();return float(total.detach())

    def checkpoint(name,k=None):
        prediction=predict(k);state={a:b.detach().cpu().clone() for a,b in net.state_dict().items()}
        path=dest/(name+'.pt');torch.save(dict(state=state,targets=targets,preprocessor=pre,dimensions=dims,seed=seed,signature=sig),path)
        loaded=torch.load(path,map_location='cpu',weights_only=False);net.load_state_dict(loaded['state'])
        np.testing.assert_allclose(prediction,predict(k),rtol=1e-5,atol=1e-5,equal_nan=True)
        return prediction

    snapshots=[]
    for step in range(1,P['steps']+1):
        k=int(rng.choice(n,p=prob));updates[k]+=1;loss=step_task(k,opt);scheduler.step()
        if step in P['snapshot_steps']:snapshots.append(checkpoint(f'shared_{step}'))
        if step%2000==0 or step==P['steps']:
            base.emit('v25_shared',seed=seed,partition=str(dest.relative_to(OUT)),step=step,loss=loss,seconds=round(time.time()-t0))
    shared=np.mean(snapshots,0);assert np.isfinite(shared).all()
    # Every task starts from the SAME shared state and a fresh optimizer.
    initial={a:b.detach().cpu().clone() for a,b in net.state_dict().items()}
    adapted=shared.copy();specialist_updates={}
    for k in selected:
        rng=np.random.default_rng(seed+100003*(k+1));torch.manual_seed(seed+100003*(k+1))
        net.load_state_dict(initial);steps=int(np.clip(np.ceil(8*len(train_ix[k])/P['batch_size']),120,400))
        optimizer=torch.optim.AdamW(net.parameters(),lr=P['specialist_learning_rate'],weight_decay=P['weight_decay'])
        decay=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=1e-5)
        snap_steps=sorted(set(int(steps*x) for x in P['specialist_snapshots']));parts=[]
        for step in range(1,steps+1):
            step_task(k,optimizer);decay.step()
            if step in snap_steps:parts.append(checkpoint(f'task{k}_{step}',k))
        jj=np.flatnonzero(d['tid'][query]==k);specialist=np.mean(np.stack(parts)[:,jj],0)
        # Keep each variant on the same output scale before averaging across seeds.
        # Rank fusion is performed later at the complete ensemble boundary.
        adapted[jj]=(1-P['specialist_fusion_weight'])*shared[jj]+P['specialist_fusion_weight']*specialist
        specialist_updates[d['tasks'][k]]=steps
    predictions=np.stack([shared,adapted]);assert np.isfinite(predictions).all()
    np.savez_compressed(cache,fit=fit,query=query,p=predictions,selected=selected,signature=sig)
    base.save(dest/'metadata.json',dict(signature=sig,seed=seed,fit_rows=len(fit),query_rows=len(query),
         shared_updates=dict(zip(d['tasks'],updates.tolist())),specialist_updates=specialist_updates,
         seconds=time.time()-t0,model_reload_passed=True,fit_group_exclusion_checked=bool(np.any(d['split'][query]!=2)),
         test_labels_used=False,local_training=False))
    base.emit('v25_fit_complete',seed=seed,partition=str(dest.relative_to(OUT)),seconds=round(time.time()-t0))
    del net,xs,yy;torch.cuda.empty_cache()
    return predictions
