"""Metric-aligned continuous-task adaptation of partition-isolated directed bond networks."""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(_k,'2')
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import admet_v24_models as b
import admet_v27_models as gnn
import admet_v46_run as adapter
import admet_v51_run as v51

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v54_graph'
P=json.loads((ROOT/'admet_v54_graph_protocol.json').read_text(encoding='utf-8'))


def signature():
    return hashlib.sha256((gnn.signature()+b.sha(ROOT/'admet_v54_graph.py')+b.sha(ROOT/'admet_v54_graph_protocol.json')).encode()).hexdigest()


def fit(d,g,k,rows,query,partition,dest):
    dest.mkdir(parents=True,exist_ok=True);path=dest/'predictions.npz';sig=signature()
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],rows);np.testing.assert_array_equal(z['query'],query);return z['p'].copy()
    assert np.all(d['split'][rows]<2) and not set(rows)&set(query)
    if partition!='final':assert not set(d['rg'][rows])&set(d['rg'][query])
    start=time.time();parts=[];metric=d['metrics'][k]
    for seed in P['seeds']:
        source=gnn.OUT/partition/str(seed);z=np.load(source/'predictions.npz')
        if partition!='final':assert not set(d['rg'][z['fit']])&set(d['rg'][query])
        ck=torch.load(source/'checkpoint_5500.pt',map_location='cpu',weights_only=False)
        assert ck['signature']==gnn.signature()
        torch.manual_seed(seed+54000+k);rng=np.random.default_rng(seed+54000+k)
        net=gnn.BondNet(*ck['dimensions']).cuda();net.load_state_dict(ck['state'])
        x=adapter.feature_matrix(d,ck,partition)
        fitmol=np.unique(d['mi'][rows]);y=np.array([np.median(d['y'][rows[d['mi'][rows]==mi]]) for mi in fitmol])
        target=ck['target'][k];truth=target.encode(y);teacher=adapter.logits(net,g,x,fitmol)[:,:,k]
        truth=torch.tensor(truth,dtype=torch.float32,device='cuda');teacher=torch.tensor(teacher,device='cuda')
        heads=list(net.graph_heads.parameters())+list(net.hybrid_heads.parameters());headids={id(p) for p in heads}
        encoder=[p for p in net.parameters() if id(p) not in headids]
        opt=torch.optim.AdamW([{'params':encoder,'lr':P['encoder_lr']},{'params':heads,'lr':P['head_lr']}],weight_decay=.015)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,P['steps'],eta_min=5e-6);snaps=[]
        for step in range(1,P['steps']+1):
            jj=rng.choice(len(fitmol),size=min(len(fitmol),P['batch_size']),replace=True);mm=fitmol[jj]
            net.train();pred=net(gnn.pack(g,mm),x[mm])[:,:,k];true=truth[jj]
            if metric=='MAE':loss=(pred-true).abs().mean()
            else:
                loss=nn.functional.smooth_l1_loss(pred,true.expand_as(pred),beta=.4)
                diff=true[:,None]-true[None,:];mask=diff.abs()>.1
                if mask.any():loss=loss+P['rank_coefficient']*nn.functional.softplus(-diff[mask].sign()*(pred[:,:,None]-pred[:,None,:])[:,mask]).mean()
            loss=loss+P['distillation']*(pred-teacher[:,jj]).square().mean()
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step();sched.step()
            if step in P['snapshots']:
                pp=adapter.logits(net,g,x,d['mi'][query])[:,:,k]
                # Spearman keeps raw continuous latent scores; no knot interpolation or clipping.
                if metric=='MAE':pp=target.decode(pp)
                snaps.append(pp)
                file=dest/f'{seed}_{step}.pt';torch.save(dict(state=net.state_dict(),signature=sig,
                    dimensions=ck['dimensions'],target=target,preprocessor=ck['preprocessor'],task=k),file)
                saved=torch.load(file,map_location='cpu',weights_only=False);net.load_state_dict(saved['state'])
                again=adapter.logits(net,g,x,d['mi'][query])[:,:,k]
                if metric=='MAE':again=target.decode(again)
                np.testing.assert_allclose(pp,again,atol=3e-5,rtol=3e-5)
            if step%150==0:b.emit('v54_adapt',task=d['tasks'][k],partition=partition,seed=seed,step=step,seconds=round(time.time()-start))
        parts.append(np.mean(snaps,0));del net,opt;torch.cuda.empty_cache()
    p=np.mean(parts,0);np.savez_compressed(path,p=p,fit=rows,query=query,signature=sig)
    b.save(dest/'metadata.json',dict(signature=sig,fit_rows=len(rows),query_rows=len(query),scaffold_overlap=0,
        source_query_groups_excluded=True,local_gpu_training=False,test_labels_used=False,model_reload_passed=True,seconds=time.time()-start))
    return p


def variants(p,m):
    a=p if m=='MAE' else np.stack([b.rank(x) for x in p])
    return np.concatenate([p,np.mean(a,0)[None]])


def main():
    b.remote_only();OUT.mkdir(exist_ok=True);d=b.build_data();g=gnn.graphs(d)
    basev,basec=v51.baseline(d);folds=np.load(b.OUT/'folds.npy');records=[]
    for task in P['tasks']:
        k=d['tasks'].index(task);m=d['metrics'][k];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        cp=np.full((3,len(d['y'])),np.nan)
        for f in range(3):
            q=tr[folds[tr]==f];rr=tr[folds[tr]!=f]
            cp[:,q]=variants(fit(d,g,k,rr,q,'cv/fold'+str(f),OUT/task/f'fold{f}'),m)
        np.save(OUT/task/'cv_predictions.npy',cp)
        vp=variants(fit(d,g,k,tr,vi,'valid',OUT/task/'valid'),m);grid=[]
        for branch in range(3):
            for w in P['weights']:
                gs=[b.delta(m,d['y'][ix],basec[ix],b.blend(m,basec[ix],cp[branch,ix],w)) for f in range(3) for ix in [tr[folds[tr]==f]]]
                vg=b.delta(m,d['y'][vi],basev[vi],b.blend(m,basev[vi],vp[branch],w))
                grid.append(dict(variant=branch,weight=w,cv_gains=gs,cv_mean=float(np.mean(gs)),valid_gain=vg))
        held=[]
        for f in range(3):
            s=max(grid,key=lambda x:np.mean([v for j,v in enumerate(x['cv_gains']) if j!=f]));held.append(s['cv_gains'][f])
        good=[s for s in grid if s['cv_mean']>.001 and sum(v>0 for v in s['cv_gains'])>=2 and s['valid_gain']>-.03]
        selected=max(good,key=lambda x:.9*x['cv_mean']+.1*x['valid_gain']) if good and np.mean(held)>.001 else None
        records.append(dict(task=task,metric=m,selection=selected,grid=grid,held_gains=held,held_mean=float(np.mean(held))))
        b.save(OUT/'analysis.json',dict(status='IN_PROGRESS',signature=signature(),tasks=records))
        b.emit('v54_selection',task=task,selection=selected)
    b.save(OUT/'analysis.json',dict(status='FROZEN',signature=signature(),tasks=records,official_score=None))


if __name__=='__main__':main()
