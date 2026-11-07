"""CUDA task-specific MAE experts with nested stopping and frozen confirmation.

Original official graphs are preserved, including the documented valence exception.
No historical model is presented as exact v73 OOF; the new tree comparator is
refitted on precisely the same outer rows as the neural experts.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='4'
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import Crippen,Descriptors,Lipinski
from catboost import CatBoostRegressor
import admet_v67_data as b

ROOT=b.ROOT; OUT=ROOT/'admet_model_outputs_v74'
P=json.loads((ROOT/'admet_v74_protocol.json').read_text(encoding='utf-8'))
PARTS=('fold0','fold1','fold2','valid')

def signature():
    return hashlib.sha256((b.signature()+b.sha(__file__)+b.sha(ROOT/'admet_v74_protocol.json')).encode()).hexdigest()

def onehot(v,values):return [float(v==x) for x in values]+[float(v not in values)]
def atom(a):
    return onehot(a.GetAtomicNum(),[1,5,6,7,8,9,11,12,14,15,16,17,19,20,26,30,35,53])+onehot(a.GetTotalDegree(),list(range(6)))+onehot(a.GetFormalCharge(),[-2,-1,0,1,2])+onehot(int(a.GetChiralTag()),list(range(4)))+onehot(a.GetTotalNumHs(),list(range(5)))+onehot(int(a.GetHybridization()),[2,3,4,6,7])+[float(a.GetIsAromatic()),float(a.IsInRing()),a.GetMass()*.01]
def bond(e):
    return [float(e.GetBondType()==x) for x in [Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]]+[float(e.GetIsConjugated()),float(e.IsInRing())]+onehot(int(e.GetStereo()),list(range(6)))

def graphs(d):
    dest=OUT/'features';dest.mkdir(parents=True,exist_ok=True);path=dest/'graphs.pt'
    if path.exists():
        z=torch.load(path,map_location='cpu',weights_only=False);assert z['signature']==signature();return z
    gs=[];prior=[];exceptions=[]
    # Only selected task molecules are required; unused molecules are absent.
    needed=set(d['mi'][np.isin(d['tid'],[d['tasks'].index(t) for t in P['tasks']])].tolist())
    for i,sm in enumerate(d['smiles']):
        if i not in needed:gs.append(None);prior.append([0.,0.]);continue
        mol,mode=b.parse_molecule(sm)
        if mode!='full_sanitization':exceptions.append(dict(molecule=i,mode=mode))
        aa=np.array([atom(a) for a in mol.GetAtoms()],np.float32)
        ee=[];ss=[];tt=[];rr=[]
        for e in mol.GetBonds():
            j=len(ss);ss.extend([e.GetBeginAtomIdx(),e.GetEndAtomIdx()]);tt.extend([e.GetEndAtomIdx(),e.GetBeginAtomIdx()]);rr.extend([j+1,j]);ee.extend([bond(e)]*2)
        ea=np.array(ee,np.float32).reshape(-1,13)
        gs.append((aa,ea,np.array(ss,np.int64),np.array(tt,np.int64),np.array(rr,np.int64)))
        fragments=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=False)
        par=max(fragments,key=lambda x:(sum(a.GetAtomicNum()==6 for a in x.GetAtoms()),x.GetNumHeavyAtoms()))
        par.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(par,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,catchErrors=True)
        lp=Crippen.MolLogP(par);mw=Descriptors.MolWt(par);rb=Lipinski.NumRotatableBonds(par)
        arom=sum(a.GetIsAromatic() for a in par.GetAtoms())/max(1,par.GetNumHeavyAtoms())
        prior.append([lp,.16-.63*lp-.0062*mw+.066*rb-.74*arom])
    z=dict(signature=signature(),graphs=gs,prior=np.array(prior,np.float32))
    torch.save(z,path);b.save(dest/'audit.json',dict(molecules=len(needed),exceptions=exceptions,graph_replacement=False,external_labels=False));return z

def pack(g,mm):
    aa=[];ee=[];ss=[];tt=[];rr=[];na=[];aoff=0;eoff=0
    for mi in mm:
        a,e,s,t,r=g['graphs'][int(mi)];aa.append(a);ee.append(e);ss.append(s+aoff);tt.append(t+aoff);rr.append(r+eoff);na.append(len(a));aoff+=len(a);eoff+=len(e)
    arrays=[np.concatenate(aa),np.concatenate(ee),np.concatenate(ss),np.concatenate(tt),np.concatenate(rr),np.repeat(np.arange(len(mm)),na),np.array(na,np.float32)]
    return [torch.from_numpy(x).cuda() for x in arrays]

class Net(nn.Module):
    def __init__(self,fdim,task):
        super().__init__();h=P['hidden'];self.task=task
        self.edge=nn.Linear(52+13,h);self.message=nn.Linear(h,h,bias=False);self.atom=nn.Linear(52+h,h)
        def block(a,c):return nn.Sequential(nn.Linear(a,c),nn.LayerNorm(c),nn.SiLU(),nn.Dropout(P['dropout']))
        self.pool=block(2*h+1,h);self.fp=nn.Sequential(block(fdim,256),block(256,h))
        self.gh=nn.Sequential(block(h,128),nn.Linear(128,1));self.fh=nn.Sequential(block(h,128),nn.Linear(128,1));self.hh=nn.Sequential(block(2*h,256),block(256,128),nn.Linear(128,1))
    def forward(self,g,x,prior,center,scale):
        a,e,s,t,r,mi,na=g
        h0=nn.functional.silu(self.edge(torch.cat([a[s],e],1)));h=h0
        for _ in range(P['depth']-1):
            incoming=h.new_zeros((len(a),P['hidden']));incoming.index_add_(0,t,h)
            h=nn.functional.dropout(nn.functional.silu(h0+self.message(incoming[s]-h[r])),P['dropout'],self.training)
        incoming=h.new_zeros((len(a),P['hidden']));incoming.index_add_(0,t,h)
        av=nn.functional.silu(self.atom(torch.cat([a,incoming],1)));pool=av.new_zeros((len(na),P['hidden']));pool.index_add_(0,mi,av)
        gh=self.pool(torch.cat([pool/na[:,None],pool/na.sqrt()[:,None],na.log1p()[:,None]],1));fh=self.fp(nn.functional.dropout(x,.04,self.training))
        z=torch.stack([self.gh(gh).squeeze(-1),self.fh(fh).squeeze(-1),self.hh(torch.cat([gh,fh],1)).squeeze(-1)])
        if self.task=='ppbr_az':return 100*torch.sigmoid(z+2.)
        return z*scale+prior[None] if self.task in ('lipophilicity_astrazeneca','solubility_aqsoldb') else z*scale+center

def rows(d,k,part):
    active=d['tid']==k
    if part=='final':fit=np.flatnonzero(active&(d['split']<2));q=np.flatnonzero(active&(d['split']==2))
    elif part=='valid':fit=np.flatnonzero(active&(d['split']==0));q=np.flatnonzero(active&(d['split']==1))
    else:
        f=np.load(b.OUT/'folds.npy');j=int(part[-1]);fit=np.flatnonzero(active&(d['split']==0)&(f!=j));q=np.flatnonzero(active&(d['split']==0)&(f==j))
    b.assert_excluded(d,fit,q);return fit,q

def inner(d,fit,k):
    groups=np.unique(d['rg'][fit]);rng=np.random.default_rng(P['inner_seed']+k)
    held=rng.permutation(groups)[:max(1,round(len(groups)*P['inner_group_fraction']))]
    active=np.isin(d['rg'][fit],held);tr,va=fit[~active],fit[active];b.assert_excluded(d,tr,va);return tr,va

def design(d,g,fit):
    pre=b.preprocess(d,fit);x=b.transform(d['x'],pre)
    x=np.concatenate([x,g['prior']],1).astype(np.float32)
    # Priors are fit-free analytical expressions; standardize their inputs on fitting molecules.
    mm=np.unique(d['mi'][fit]);pm=x[mm,-2:].mean(0);ps=x[mm,-2:].std(0).clip(.01);x[:,-2:]=((x[:,-2:]-pm)/ps).clip(-8,8)
    return torch.tensor(x,device='cuda'),dict(pre=pre,prior_mean=pm,prior_scale=ps)

def prior(g,d,k,ix):
    task=d['tasks'][k];col=0 if task=='lipophilicity_astrazeneca' else 1 if task=='solubility_aqsoldb' else None
    return np.zeros(len(ix),np.float32) if col is None else g['prior'][d['mi'][ix],col]

def predict(net,d,g,x,q,k,center,scale,batch=96):
    net.eval();parts=[];pr=prior(g,d,k,q)
    with torch.inference_mode():
        for s in range(0,len(q),batch):
            qq=q[s:s+batch];mm=d['mi'][qq]
            pp=net(pack(g,mm),x[mm],torch.tensor(pr[s:s+batch],device='cuda'),center,scale)
            parts.append(pp.cpu().numpy().astype(float))
    p=np.concatenate(parts,1);assert np.isfinite(p).all();return p

def train_net(d,g,k,fit,q,seed,dest,epochs=None):
    dest.mkdir(parents=True,exist_ok=True);path=dest/'predictions.npz'
    if path.exists():
        z=np.load(path);assert str(z['signature'])==signature();np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q);return z['p'].copy()
    b.assert_excluded(d,fit,q);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    x,state=design(d,g,fit);center=float(np.median(d['y'][fit]));scale=max(float(np.subtract(*np.quantile(d['y'][fit],[.75,.25]))),.1)
    net=Net(x.shape[1],d['tasks'][k]).cuda();opt=torch.optim.AdamW(net.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay'])
    total=P['epochs'];sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total,eta_min=1e-5)
    _,inv,cnt=np.unique(d['canonical'][d['mi'][fit]],return_inverse=True,return_counts=True);sw=1/cnt[inv]
    y=torch.tensor(d['y'][fit],dtype=torch.float32,device='cuda');sw=torch.tensor(sw,dtype=torch.float32,device='cuda');pr=torch.tensor(prior(g,d,k,fit),device='cuda')
    chosen=[None]*3;best=np.full(3,np.inf);bestepoch=np.zeros(3,int);started=time.time()
    head_epochs=None if epochs is None else np.broadcast_to(np.asarray(epochs,int),(3,)).copy()
    maxepoch=total if epochs is None else int(max(head_epochs))
    for epoch in range(1,maxepoch+1):
        order=rng.permutation(len(fit));net.train()
        for s in range(0,len(order),P['batch_size']):
            jj=order[s:s+P['batch_size']];mm=d['mi'][fit[jj]]
            p=net(pack(g,mm),x[mm],pr[jj],center,scale);err=(p-y[jj][None])/scale
            loss=(.95*err.abs()+.05*nn.functional.smooth_l1_loss(err,torch.zeros_like(err),beta=.2,reduction='none'))
            loss=(loss*sw[jj][None]).sum()/sw[jj].sum()/3
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step()
        sched.step()
        if head_epochs is not None:
            for j in range(3):
                if epoch==head_epochs[j]:
                    chosen[j]={key:v.detach().cpu().clone() for key,v in net.state_dict().items()};bestepoch[j]=epoch
        if epochs is None and epoch%5==0:
            p=predict(net,d,g,x,q,k,center,scale);scores=np.mean(abs(p-d['y'][q][None]),axis=1)
            state_cpu=None
            for j in range(3):
                if scores[j]<best[j]-1e-5:
                    best[j]=scores[j];bestepoch[j]=epoch
                    if state_cpu is None:state_cpu={key:v.detach().cpu().clone() for key,v in net.state_dict().items()}
                    chosen[j]=state_cpu
            if epoch>=50 and epoch-max(bestepoch)>=P['patience_epochs']:break
        if epoch%30==0:b.emit('v74_neural',task=d['tasks'][k],partition=str(dest.relative_to(OUT)),epoch=epoch,seconds=round(time.time()-started),inner_best=best.tolist() if epochs is None else None)
    ps=[]
    for j,st in enumerate(chosen):
        net.load_state_dict(st);p=predict(net,d,g,x,q,k,center,scale)
        ps.append(p[j]);torch.save(dict(signature=signature(),state=st,design=state,center=center,scale=scale,head=j,task=d['tasks'][k],fdim=x.shape[1],epochs=int(bestepoch[j]),seed=seed),dest/f'head{j}.pt')
    p=np.stack(ps);np.savez_compressed(path,p=p,fit=fit,query=q,signature=signature())
    b.save(dest/'metadata.json',dict(signature=signature(),epochs=bestepoch.tolist(),seed=seed,fit_rows=len(fit),query_rows=len(q),inner_stopping=epochs is None,seconds=time.time()-started,test_labels_used=False))
    del net,opt,x;torch.cuda.empty_cache();return p

def train_tree(d,k,fit,q,dest,trees=None):
    dest.mkdir(parents=True,exist_ok=True);path=dest/'predictions.npz'
    if path.exists():
        z=np.load(path);assert str(z['signature'])==signature();np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q);return z['p'].copy()
    b.assert_excluded(d,fit,q);pre=b.preprocess(d,fit);x=b.transform(d['x'],pre);tf=b.target(d['y'][fit],'MAE')
    model=CatBoostRegressor(iterations=2500 if trees is None else int(trees),depth=6,learning_rate=.035,l2_leaf_reg=12,loss_function='MAE',eval_metric='MAE',task_type='GPU',devices='0',random_seed=P['development_seed'],thread_count=4,verbose=False,allow_writing_files=False,border_count=128,bootstrap_type='Bernoulli',subsample=.85)
    _,inv,cnt=np.unique(d['canonical'][d['mi'][fit]],return_inverse=True,return_counts=True);sw=1/cnt[inv]
    args=dict(sample_weight=sw)
    if trees is None:args.update(eval_set=(x[d['mi'][q]],b.encode(d['y'][q],tf)),use_best_model=True,early_stopping_rounds=250)
    model.fit(x[d['mi'][fit]],b.encode(d['y'][fit],tf),**args)
    p=model.predict(x[d['mi'][q]])*tf['scale']+tf['center']
    if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
    model.save_model(str(dest/'model.cbm'));np.savez_compressed(path,p=p,fit=fit,query=q,signature=signature())
    torch.save(dict(pre=pre,target=tf),dest/'state.pt');b.save(dest/'metadata.json',dict(signature=signature(),trees=model.tree_count_,fit_rows=len(fit),query_rows=len(q),inner_stopping=trees is None));return p

def develop(d,g,tasks,parts):
    for task in tasks:
        k=d['tasks'].index(task)
        for part in parts:
            fit,q=rows(d,k,part);tr,va=inner(d,fit,k);dest=OUT/task/part
            train_net(d,g,k,tr,va,P['development_seed'],dest/'inner')
            meta=json.loads((dest/'inner/metadata.json').read_text());epochs=meta['epochs']
            p=train_net(d,g,k,fit,q,P['development_seed'],dest/'outer',epochs)
            train_tree(d,k,tr,va,dest/'tree_inner');trees=json.loads((dest/'tree_inner/metadata.json').read_text())['trees']
            tp=train_tree(d,k,fit,q,dest/'tree',trees)
            b.emit('v74_outer',task=task,part=part,epochs=epochs,tree_mae=b.metric_score('MAE',d['y'][q],tp),neural_mae=[b.metric_score('MAE',d['y'][q],a) for a in p])
    b.save(OUT/('completed_'+'_'.join(parts)+'.json'),dict(status='COMPLETE',signature=signature(),tasks=tasks,parts=parts))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','features']);ap.add_argument('--tasks',nargs='+',default=P['tasks']);ap.add_argument('--parts',nargs='+',default=PARTS);a=ap.parse_args()
    assert os.name!='nt' and torch.cuda.is_available(),'Use authorized remote CUDA server'
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    OUT.mkdir(exist_ok=True);b.save(OUT/'protocol.json',dict(signature=signature(),protocol=P));d=b.build();g=graphs(d)
    assert g['graphs'][int(d['mi'][np.flatnonzero(d['tid']==d['tasks'].index(P['tasks'][0]))[0]])][0].shape[1]==52
    if a.stage=='develop':develop(d,g,a.tasks,a.parts)

if __name__=='__main__':main()
