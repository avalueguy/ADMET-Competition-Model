"""Remote task specialists: differentiable property priors and bounded PPBR outputs."""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(_k,'2')
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import admet_v24_models as b
import admet_v51_run as v51

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_model_outputs_v52_neural'
SEEDS=[521016,521017]
EPOCHS=140


def signature():
    return hashlib.sha256((b.sha(ROOT/'admet_v52_neural.py')+v51.signature()).encode()).hexdigest()


class Net(nn.Module):
    def __init__(self,fpdim,densedim,family,task):
        super().__init__();self.family=family;self.task=task
        def layer(a,c):return nn.Sequential(nn.Linear(a,c),nn.LayerNorm(c),nn.SiLU(),nn.Dropout(.18))
        self.fp=layer(fpdim,256);self.dense=layer(densedim,128)
        self.body=nn.Sequential(layer(384,256),layer(256,128),nn.Linear(128,1))

    def forward(self,fp,dense,prior,center,scale):
        if self.training:fp=nn.functional.dropout(fp,.06,True)
        z=self.body(torch.cat([self.fp(fp),self.dense(dense)],1)).squeeze(-1)
        if self.family==0:return z*scale+center
        if self.task=='ppbr_az':return 100*torch.sigmoid(z+2.)
        return z*scale+prior


def fit(d,c,k,rows,query,dest):
    dest.mkdir(parents=True,exist_ok=True);path=dest/'predictions.npz';sig=signature()
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],rows);np.testing.assert_array_equal(z['query'],query);return z['p'].copy()
    assert not set(rows)&set(query) and np.all(d['split'][rows]<2)
    if np.any(d['split'][query]<2):assert not set(d['rg'][rows])&set(d['rg'][query])
    ix=np.unique(d['mi'][np.r_[rows,query]]);lookup=np.full(len(d['smiles']),-1);lookup[ix]=np.arange(len(ix))
    fit_ix=lookup[d['mi'][rows]];query_ix=lookup[d['mi'][query]]
    raw=np.concatenate([d['desc'],d['emb'],c['prior']],1)
    pre=b.Preprocessor().fit(raw[np.unique(d['mi'][rows])]);dense=pre.transform(raw[ix])
    # Full original ECFP/FCFP views; no variance filtering of sparse rare motifs.
    fp=np.concatenate([d['fp2'][ix],d['fcfp'][ix]],1)
    name=d['tasks'][k];pp=c['prior'][ix,0] if name=='lipophilicity_astrazeneca' else (c['prior'][ix,1] if name=='solubility_aqsoldb' else np.zeros(len(ix)))
    tensors=[torch.tensor(x,dtype=torch.float32,device='cuda') for x in [fp,dense,pp]]
    y=torch.tensor(d['y'][rows],dtype=torch.float32,device='cuda')
    _,inv,count=np.unique(d['canonical'][d['mi'][rows]],return_inverse=True,return_counts=True)
    weight=torch.tensor(1./count[inv],dtype=torch.float32,device='cuda')
    center=float(np.median(d['y'][rows]));scale=max(.1,float(np.subtract(*np.quantile(d['y'][rows],[.75,.25]))))
    parts=[];start=time.time()
    for seed in SEEDS:
        families=[]
        for family in range(2):
            torch.manual_seed(seed+family*100);rng=np.random.default_rng(seed+family*100)
            net=Net(fp.shape[1],dense.shape[1],family,name).cuda()
            opt=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.02)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-5)
            snaps=[]
            def predict():
                net.eval()
                with torch.inference_mode():
                    return np.concatenate([net(*[x[query_ix[s:s+512]] for x in tensors],center,scale).cpu().numpy() for s in range(0,len(query),512)])
            for epoch in range(1,EPOCHS+1):
                order=rng.permutation(len(rows));net.train()
                for s in range(0,len(order),192):
                    jj=order[s:s+192];mi=fit_ix[jj]
                    pred=net(*[x[mi] for x in tensors],center,scale)
                    losses=(pred-y[jj]).abs()/scale
                    loss=(losses*weight[jj]).sum()/weight[jj].sum()
                    assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step()
                schedule.step()
                if epoch in [90,115,140]:
                    p=predict();snaps.append(p)
                    file=dest/f'family{family}_{seed}_{epoch}.pt'
                    torch.save(dict(state=net.state_dict(),dimensions=[fp.shape[1],dense.shape[1]],family=family,
                        task=name,pre=pre,center=center,scale=scale,signature=sig),file)
                    ck=torch.load(file,map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
                    np.testing.assert_allclose(p,predict(),atol=1e-5,rtol=1e-5)
                if epoch%35==0:b.emit('v52_neural',task=name,partition=dest.name,seed=seed,family=family,epoch=epoch,seconds=round(time.time()-start))
            families.append(np.mean(snaps,0));del net,opt
        parts.append(np.stack(families));torch.cuda.empty_cache()
    p=np.mean(parts,0);np.savez_compressed(path,p=p,fit=rows,query=query,signature=sig)
    b.save(dest/'metadata.json',dict(signature=sig,fit_rows=len(rows),query_rows=len(query),seconds=time.time()-start,
        local_gpu_training=False,model_reload_passed=True,test_labels_used=False))
    return p


def variants(p):return np.concatenate([p,np.mean(p,0)[None]])


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',nargs='+',default=['solubility_aqsoldb','lipophilicity_astrazeneca','ppbr_az']);a=p.parse_args()
    b.remote_only();OUT.mkdir(exist_ok=True);d=b.build_data();c=v51.chemistry(d)
    f=np.load(b.OUT/'folds.npy');base=np.load(ROOT/'admet_model_outputs_v46/baseline_cv_proxy.npy');records=[]
    for t in a.tasks:
        k=d['tasks'].index(t);tr=np.flatnonzero((d['tid']==k)&(d['split']==0));cp=np.full((3,len(d['y'])),np.nan)
        for fold in range(3):
            rr=tr[f[tr]!=fold];q=tr[f[tr]==fold];cp[:,q]=variants(fit(d,c,k,rr,q,OUT/t/f'fold{fold}'))
        np.save(OUT/t/'cv_predictions.npy',cp)
        grid=[]
        for branch in range(3):
            for weight in [.15,.35,.6,.85,1.]:
                gs=[b.delta('MAE',d['y'][ix],base[ix],b.blend('MAE',base[ix],cp[branch,ix],weight)) for fold in range(3) for ix in [tr[f[tr]==fold]]]
                grid.append(dict(variant=branch,weight=weight,cv_gains=gs,cv_mean=float(np.mean(gs))))
        held=[]
        for fold in range(3):
            z=max(grid,key=lambda z:np.mean([g for j,g in enumerate(z['cv_gains']) if j!=fold]));held.append(z['cv_gains'][fold])
        good=[z for z in grid if z['cv_mean']>.001 and sum(g>0 for g in z['cv_gains'])>=2]
        selected=max(good,key=lambda z:z['cv_mean']) if good and np.mean(held)>.001 else None
        records.append(dict(task=t,selection=selected,grid=grid,held_gains=held,held_mean=float(np.mean(held))))
        b.save(OUT/'analysis.json',dict(status='IN_PROGRESS',signature=signature(),tasks=records))
        b.emit('v52_selection',task=t,selection=selected,held_mean=float(np.mean(held)))
    b.save(OUT/'analysis.json',dict(status='FROZEN',signature=signature(),tasks=records,official_score_prediction=None))


if __name__=='__main__':main()
