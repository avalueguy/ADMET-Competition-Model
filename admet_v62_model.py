"""Remote residual atom GINE with virtual nodes and task-dependent multi-view gating."""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[_k]='2'
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
import admet_v24_models as b
import admet_v27_models as graph

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v62'
P=json.loads((ROOT/'admet_v62_protocol.json').read_text(encoding='utf-8'))

def signature():return hashlib.sha256((b.sha(__file__)+b.sha(ROOT/'admet_v62_protocol.json')+b.signature()).encode()).hexdigest()

class Layer(nn.Module):
    def __init__(self,h,bond):
        super().__init__();self.edge=nn.Linear(bond,h);self.eps=nn.Parameter(torch.zeros(1));self.mlp=nn.Sequential(nn.Linear(h,2*h),nn.SiLU(),nn.Dropout(P['dropout']),nn.Linear(2*h,h));self.norm=nn.LayerNorm(h)
        self.virtual=nn.Sequential(nn.Linear(2*h,h),nn.SiLU(),nn.Linear(h,h));self.vn=nn.LayerNorm(h)
    def forward(self,x,e,src,dst,mol,natoms,v):
        msg=nn.functional.silu(x[src]+self.edge(e));agg=x.new_zeros(x.shape);agg.index_add_(0,dst,msg)
        degree=torch.bincount(dst,minlength=len(x)).to(x.dtype).clamp_min(1).sqrt()[:,None]
        update=self.mlp((1+self.eps)*x+agg/degree+v[mol]);x=self.norm(x+nn.functional.dropout(update,P['dropout'],self.training))
        pooled=x.new_zeros(v.shape);pooled.index_add_(0,mol,x);pooled/=natoms[:,None]
        v=self.vn(v+self.virtual(torch.cat([v,pooled],1)));return x,v

class Net(nn.Module):
    def __init__(self,atomdim,bonddim,featdim,ntasks):
        super().__init__();h=P['hidden'];self.atom=nn.Linear(atomdim,h);self.virtual=nn.Parameter(torch.zeros(1,h));self.layers=nn.ModuleList([Layer(h,bonddim) for _ in range(P['depth'])]);self.attention=nn.Linear(h,1)
        self.graph=nn.Sequential(nn.Linear(4*h,h),nn.LayerNorm(h),nn.SiLU(),nn.Dropout(P['dropout']))
        self.chem=nn.Sequential(nn.Linear(featdim,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(.25),nn.Linear(256,h),nn.LayerNorm(h),nn.SiLU())
        self.task=nn.Embedding(ntasks,24);self.gate=nn.Sequential(nn.Linear(2*h+24,64),nn.SiLU(),nn.Linear(64,1))
        self.graphheads=nn.ModuleList([nn.Sequential(nn.Linear(h,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,1)) for _ in range(ntasks)])
        self.hybridheads=nn.ModuleList([nn.Sequential(nn.Linear(2*h,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,1)) for _ in range(ntasks)])
    def forward(self,packed,features):
        atom,bond,src,dst,rev,mol,natoms=packed;x=nn.functional.silu(self.atom(atom));v=self.virtual.expand(len(natoms),-1);means=[]
        for layer in self.layers:
            x,v=layer(x,bond,src,dst,mol,natoms,v);mean=x.new_zeros((len(natoms),x.shape[1]));mean.index_add_(0,mol,x);means.append(mean/natoms[:,None])
        maximum=x.new_full((len(natoms),x.shape[1]),-torch.inf);maximum.scatter_reduce_(0,mol[:,None].expand_as(x),x,reduce='amax',include_self=True)
        scores=self.attention(x).squeeze(-1);smax=scores.new_full((len(natoms),),-torch.inf);smax.scatter_reduce_(0,mol,scores,reduce='amax',include_self=True)
        weight=(scores-smax[mol]).exp();den=weight.new_zeros(len(natoms));den.index_add_(0,mol,weight);att=x.new_zeros(v.shape);att.index_add_(0,mol,x*(weight/den[mol].clamp_min(1e-8))[:,None])
        gh=self.graph(torch.cat([torch.stack(means).mean(0),maximum,att,v],1));ch=self.chem(nn.functional.dropout(features,.04,self.training));a=[];c=[]
        for k in range(len(self.graphheads)):
            t=self.task.weight[k].expand(len(gh),-1);gate=self.gate(torch.cat([gh,ch,t],1)).sigmoid();a.append(self.graphheads[k](gh).squeeze(-1));c.append(self.hybridheads[k](torch.cat([gh*gate,ch*(1-gate)],1)).squeeze(-1))
        return torch.stack([torch.stack(a,1),torch.stack(c,1)])

def prepare(d,rows):
    nt=len(d['tasks']);targets=[];y=np.zeros((len(d['smiles']),nt),np.float32);mask=np.zeros_like(y,bool);indices=[]
    for k in range(nt):
        ix=rows[d['tid'][rows]==k];tf=b.Target(d['y'][ix],d['metrics'][k]);targets.append(tf);mols=d['mi'][ix];unique=np.unique(mols);indices.append(unique)
        for mi in unique:
            yy=d['y'][ix[mols==mi]];value=np.mean(yy) if d['metrics'][k] in ['AUROC','AUPRC'] else np.median(yy);y[mi,k]=tf.encode(np.array([value]))[0];mask[mi,k]=True
    unique=np.unique(d['mi'][rows]);raw=np.concatenate([d['desc'],d['emb']],1);pre=b.Preprocessor().fit(raw[unique]);features=np.concatenate([d['fp2'],d['fcfp'],pre.transform(raw)],1)
    return targets,indices,unique,pre,torch.tensor(features,device='cuda'),torch.tensor(y,device='cuda'),torch.tensor(mask,device='cuda')

def predict(net,d,g,features,query,targets):
    net.eval();mm=np.unique(d['mi'][query]);values=np.empty((2,len(d['smiles']),len(d['tasks'])),np.float32)
    with torch.inference_mode():
        for start in range(0,len(mm),96):
            mols=mm[start:start+96];values[:,mols]=net(graph.pack(g,mols),features[mols]).cpu().numpy()
    out=np.empty((2,len(query)))
    for k,m in enumerate(d['metrics']):
        ix=np.flatnonzero(d['tid'][query]==k);p=values[:,d['mi'][query[ix]],k];out[:,ix]=p if m=='Spearman' else targets[k].decode(p)
    assert np.isfinite(out).all();return out

def fit(d,g,rows,query,seed,dest):
    assert os.name!='nt' and ROOT==Path(P['remote_root']) and torch.cuda.is_available();dest.mkdir(parents=True,exist_ok=True);sig=signature();path=dest/'predictions.npz'
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig;np.testing.assert_array_equal(rows,z['fit']);np.testing.assert_array_equal(query,z['query']);return z['p'].copy()
    assert not set(rows)&set(query) and np.all(d['split'][rows]<2)
    if np.any(d['split'][query]<2):
        assert not set(d['rg'][rows])&set(d['rg'][query]);assert not set(d['canonical'][d['mi'][rows]])&set(d['canonical'][d['mi'][query]])
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);targets,indices,unique,pre,features,y,observed=prepare(d,rows)
    dims=[g['atom'].shape[1],g['bond'].shape[1],features.shape[1],len(d['tasks'])];net=Net(*dims).cuda();opt=torch.optim.AdamW(net.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,P['steps'],eta_min=P['minimum_learning_rate']);snap=[];start=time.time();updates=np.zeros(len(d['tasks']),int)
    for step in range(1,P['steps']+1):
        net.train();k=int(rng.integers(len(indices)));n=P['batch_size']//2;mm=np.r_[rng.choice(indices[k],min(n,len(indices[k])),replace=True),rng.choice(unique,n,replace=True)];p=net(graph.pack(g,mm),features[mm]);by=y[mm];bm=observed[mm];losses=[]
        for j,m in enumerate(d['metrics']):
            active=bm[:,j]
            if not active.any():continue
            updates[j]+=1;true=by[active,j];pp=p[:,active,j]
            if m in ['AUROC','AUPRC']:
                loss=nn.functional.binary_cross_entropy_with_logits(pp,true.expand_as(pp));pos=pp[:,true>.5];neg=pp[:,true<.5]
                if pos.shape[1] and neg.shape[1]:loss+=.12*nn.functional.softplus(neg[:,None]-pos[:,:,None]).mean()
            elif m=='MAE':loss=.9*(pp-true).abs().mean()+.1*nn.functional.smooth_l1_loss(pp,true.expand_as(pp),beta=.3)
            else:
                loss=nn.functional.smooth_l1_loss(pp,true.expand_as(pp),beta=.35);diff=true[:,None]-true[None,:];ok=diff.abs()>.15
                if ok.any():loss+=.15*nn.functional.softplus(-diff[ok].sign()*(pp[:,:,None]-pp[:,None,:])[:,ok]).mean()
            losses.append(loss)
        loss=torch.stack(losses).mean();assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step();sched.step()
        if step in P['snapshot_steps']:
            pp=predict(net,d,g,features,query,targets);snap.append(pp);file=dest/f'checkpoint_{step}.pt';torch.save(dict(state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},target=targets,pre=pre,dimensions=dims,signature=sig,seed=seed,step=step),file);saved=torch.load(file,map_location='cpu',weights_only=False);net.load_state_dict(saved['state']);np.testing.assert_allclose(pp,predict(net,d,g,features,query,targets),atol=3e-5,rtol=3e-5)
        if step%500==0:b.emit('v62_train',partition=str(dest.relative_to(OUT)),step=step,seed=seed,seconds=round(time.time()-start),loss=float(loss.detach()))
    pp=np.mean(snap,0);np.savez_compressed(path,p=pp,fit=rows,query=query,signature=sig);b.save(dest/'metadata.json',dict(signature=sig,seed=seed,fit_rows=len(rows),query_rows=len(query),updates=updates.tolist(),scaffold_overlap=0 if np.any(d['split'][query]<2) else None,model_reload_passed=True,test_labels_used=False,local_gpu_training=False,seconds=time.time()-start));del net,opt,features,y,observed;torch.cuda.empty_cache();return pp

def selftest(d,g):
    torch.manual_seed(621017);net=Net(g['atom'].shape[1],g['bond'].shape[1],10,len(d['tasks'])).cuda().eval();x=torch.zeros((4,10),device='cuda')
    with torch.inference_mode():
        p=net(graph.pack(g,[0,1,2]),x[:3]).cpu().numpy();q=net(graph.pack(g,[2,0,1]),x[:3]).cpu().numpy();z=net(graph.pack(g,[0]),x[:1]).cpu().numpy()
    np.testing.assert_allclose(p[:,[2,0,1]],q,atol=3e-5,rtol=3e-5);np.testing.assert_allclose(p[:,:1],z,atol=3e-5,rtol=3e-5);b.save(OUT/'selftest.json',dict(status='PASS',batch_invariance=True,molecule_order_invariance=True));del net;torch.cuda.empty_cache()
