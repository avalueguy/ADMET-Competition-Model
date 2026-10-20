"""Independent MAE specialist models on the authorized remote host."""
import json,os,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from catboost import CatBoostRegressor
import admet_v24_models as b
from admet_v25_run import v24_valid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v28'
P=json.loads((ROOT/'admet_v28_protocol.json').read_text())

def signature():
    return hashlib.sha256(json.dumps(dict(base=b.signature(),files={n:b.sha(ROOT/n) for n in ['admet_v28_run.py','admet_v28_protocol.json']}),sort_keys=True).encode()).hexdigest()

class Net(nn.Module):
    def __init__(self,dim):
        super().__init__();h=P['hidden'];dp=P['dropout']
        self.input=nn.Sequential(nn.Linear(dim,h),nn.LayerNorm(h),nn.SiLU(),nn.Dropout(dp))
        self.layers=nn.ModuleList([nn.Sequential(nn.Linear(h,h),nn.LayerNorm(h),nn.SiLU(),nn.Dropout(dp)) for _ in range(2)])
        self.head=nn.Linear(h,1)
    def forward(self,x):
        x=self.input(x)
        for layer in self.layers:x=x+layer(x)
        return self.head(x).squeeze(-1)

def fit(d,k,rows,query,dest):
    dest.mkdir(parents=True,exist_ok=True);sig=signature();cache=dest/'predictions.npz'
    if cache.exists():
        z=np.load(cache);assert str(z['signature'])==sig
        np.testing.assert_array_equal(rows,z['fit']);np.testing.assert_array_equal(query,z['query']);return z['p']
    assert not set(rows)&set(query) and np.all(d['split'][rows]<2)
    if np.any(d['split'][query]<2):assert not set(d['rg'][rows])&set(d['rg'][query])
    mi=d['mi'];unique=np.unique(mi[rows]);dense=np.concatenate([d['desc'],d['emb']],1)
    pre=b.Preprocessor().fit(dense[unique]);values=pre.transform(dense)
    raw=np.concatenate([d['fp2'],d['fcfp'],values],1);x=torch.tensor(raw[mi[rows]],device='cuda')
    qx=torch.tensor(raw[mi[query]],device='cuda');tf=b.Target(d['y'][rows],'MAE')
    y=torch.tensor(tf.encode(d['y'][rows]),device='cuda');neural=[];t0=time.time()
    for seed in P['seeds']:
        rng=np.random.default_rng(seed);torch.manual_seed(seed);net=Net(raw.shape[1]).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay'])
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,P['epochs'],eta_min=2e-5);snap=[]
        for epoch in range(1,P['epochs']+1):
            net.train();order=rng.permutation(len(rows))
            for start in range(0,len(order),P['batch_size']):
                ix=order[start:start+P['batch_size']];p=net(x[ix]);loss=(p-y[ix]).abs().mean()
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step()
            sched.step()
            if epoch in P['snapshots']:
                net.eval()
                with torch.inference_mode():pred=np.concatenate([net(qx[start:start+512]).cpu().numpy() for start in range(0,len(query),512)])
                checkpoint=dict(state={name:v.detach().cpu().clone() for name,v in net.state_dict().items()},target=tf,preprocessor=pre,signature=sig)
                path=dest/f'mlp_{seed}_{epoch}.pt';torch.save(checkpoint,path);net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state'])
                with torch.inference_mode():again=np.concatenate([net(qx[start:start+512]).cpu().numpy() for start in range(0,len(query),512)])
                np.testing.assert_allclose(pred,again,rtol=1e-5,atol=1e-5);snap.append(tf.decode(pred))
        neural.append(np.mean(snap,0));del net
    del x,qx,y;torch.cuda.empty_cache();boost=[]
    for seed,depth in zip(P['seeds'],P['boost_depths']):
        # Descriptors + frozen embedding with MAE loss complement fingerprint specialists.
        model=CatBoostRegressor(iterations=P['boost_iterations'],depth=depth,learning_rate=P['boost_learning_rate'],
             l2_leaf_reg=10,loss_function='MAE',random_seed=seed,task_type='GPU',devices='0',gpu_ram_part=P['gpu_ram_part'],
             thread_count=3,border_count=128,verbose=False,allow_writing_files=False)
        model.fit(values[mi[rows]],tf.encode(d['y'][rows]));path=dest/f'boost_{seed}.cbm';model.save_model(str(path))
        pred=model.predict(values[mi[query]]);reload=CatBoostRegressor();reload.load_model(str(path))
        np.testing.assert_allclose(pred,reload.predict(values[mi[query]]),rtol=1e-8,atol=1e-8);boost.append(tf.decode(pred))
    p=np.stack([np.mean(neural,0),np.mean(boost,0)]);assert np.isfinite(p).all()
    np.savez_compressed(cache,p=p,fit=rows,query=query,signature=sig)
    b.save(dest/'metadata.json',dict(signature=sig,task=d['tasks'][k],fit_rows=len(rows),query_rows=len(query),seconds=time.time()-t0,
                                   model_reload_passed=True,local_gpu_training=False,test_labels_used=False))
    b.emit('v28_fit',task=d['tasks'][k],partition=str(dest.relative_to(OUT)),seconds=round(time.time()-t0));return p

def evaluate(d):
    folds=np.load(b.OUT/'folds.npy');old=np.load(b.OUT/'cv_predictions.npy')[1];base=v24_valid(d)
    results=[];validpred={};cvpred={}
    for t in P['tasks']:
        k=d['tasks'].index(t);assert d['metrics'][k]=='MAE'
        tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        vp=fit(d,k,tr,vi,OUT/t/'valid');cp=np.full((2,len(d['y'])),np.nan)
        for fold in range(3):
            query=tr[folds[tr]==fold];rows=tr[folds[tr]!=fold]
            cp[:,query]=fit(d,k,rows,query,OUT/t/f'fold{fold}')
        np.save(OUT/t/'cv_predictions.npy',cp);grid=[]
        for family,name in enumerate(P['families']):
            for weight in P['weights']:
                if not weight:continue
                fg=[]
                for fold in range(3):
                    ix=tr[folds[tr]==fold];fg.append(b.delta('MAE',d['y'][ix],old[ix],b.blend('MAE',old[ix],cp[family,ix],weight)))
                vg=b.delta('MAE',d['y'][vi],base[vi],b.blend('MAE',base[vi],vp[family],weight))
                obj=.6*vg+.4*np.mean(fg)-.1*np.std(fg)
                eligible=vg>1e-4 and np.mean(fg)>-.01 and sum(v>=0 for v in fg)>=1 and obj>1e-4
                grid.append(dict(family=family,name=name,weight=weight,cv_gains=fg,cv_mean=float(np.mean(fg)),valid_gain=vg,
                                 objective=float(obj),eligible=bool(eligible)))
        candidates=[r for r in grid if r['eligible']];sel=max(candidates,key=lambda r:r['objective']) if candidates else None
        record=dict(task=t,metric='MAE',selection=sel,grid=grid,valid_baseline=b.metric('MAE',d['y'][vi],base[vi]),
                    valid_candidates=[b.metric('MAE',d['y'][vi],v) for v in vp])
        results.append(record);b.save(OUT/'analysis.json',dict(signature=signature(),tasks=results,status='IN_PROGRESS',limitation=P['limitation']))
        b.emit('v28_selection',task=t,selection=sel)
    b.save(OUT/'analysis.json',dict(signature=signature(),tasks=results,status='EVALUATED',selected_tasks=[r['task'] for r in results if r['selection']],limitation=P['limitation']))

def final(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature();pred={}
    for r in report['tasks']:
        if r['selection'] is None:continue
        k=d['tasks'].index(r['task']);rows=np.flatnonzero((d['tid']==k)&(d['split']<2));query=np.flatnonzero((d['tid']==k)&(d['split']==2))
        p=fit(d,k,rows,query,OUT/r['task']/'final');pred[r['task']]=p.tolist()
    b.save(OUT/'test_predictions_compact.json',pred)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['evaluate','final','all']);args=parser.parse_args()
    b.remote_only();OUT.mkdir(parents=True,exist_ok=True);b.emit('v28_start',signature=signature());d=b.build_data()
    if args.stage in ['evaluate','all']:evaluate(d)
    if args.stage in ['final','all']:final(d)
