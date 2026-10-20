"""Remote directed bond GNN with simultaneous masked multi-task supervision."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from scipy.special import expit
import admet_v24_models as b

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_model_outputs_v27'
P=json.loads((ROOT/'admet_v27_protocol.json').read_text(encoding='utf-8'))

def signature():
    names=['admet_v27_protocol.json','admet_v27_models.py','admet_v27_run.py']
    return hashlib.sha256(json.dumps(dict(base=b.signature(),files={n:b.sha(ROOT/n) for n in names}),sort_keys=True).encode()).hexdigest()

def onehot(v,values):return [float(v==x) for x in values]+[float(v not in values)]
def atom_features(a):
    return onehot(a.GetAtomicNum(),[1,5,6,7,8,9,11,12,14,15,16,17,19,20,26,30,35,53])+onehot(a.GetTotalDegree(),list(range(6)))+onehot(a.GetFormalCharge(),[-2,-1,0,1,2])+onehot(int(a.GetChiralTag()),list(range(4)))+onehot(a.GetTotalNumHs(),list(range(5)))+onehot(int(a.GetHybridization()),[2,3,4,6,7])+[float(a.GetIsAromatic()),float(a.IsInRing()),a.GetMass()*.01]
def bond_features(e):
    return [float(e.GetBondType()==x) for x in [Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]]+[float(e.GetIsConjugated()),float(e.IsInRing())]+onehot(int(e.GetStereo()),list(range(6)))

def graphs(d):
    dest=OUT/'features';dest.mkdir(parents=True,exist_ok=True);path=dest/'graphs.npz'
    meta=dict(smiles_sha=hashlib.sha256(json.dumps(d['smiles']).encode()).hexdigest(),source_sha=b.sha(__file__))
    if path.exists():
        assert json.loads((dest/'graphs.json').read_text())==meta
        z=np.load(path);return {name:z[name] for name in z.files}
    a=[];e=[];src=[];dst=[];rev=[];ap=[0];ep=[0];invalid=0
    for sm in d['smiles']:
        mol=Chem.MolFromSmiles(sm)
        if mol is None or mol.GetNumAtoms()==0:
            invalid+=1;mol=Chem.MolFromSmiles('[He]')
        aa=[atom_features(x) for x in mol.GetAtoms()];ee=[];ss=[];tt=[];rr=[]
        for bond in mol.GetBonds():
            i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx();off=len(ss)
            ss.extend([i,j]);tt.extend([j,i]);rr.extend([off+1,off]);ee.extend([bond_features(bond)]*2)
        a.extend(aa);e.extend(ee);src.extend(ss);dst.extend(tt);rev.extend(rr)
        ap.append(len(a));ep.append(len(e))
    af=np.array(a,np.float32);bf=np.array(e,np.float32)
    z=dict(atom=af,bond=bf,src=np.array(src,np.int64),dst=np.array(dst,np.int64),rev=np.array(rev,np.int64),
           ap=np.array(ap,np.int64),ep=np.array(ep,np.int64))
    np.savez_compressed(path,**z);b.save(dest/'graphs.json',meta)
    b.emit('v27_graphs',molecules=len(d['smiles']),atoms=len(af),directed_bonds=len(bf),invalid=invalid)
    return z

def pack(g,molecules):
    atoms=[];bonds=[];sources=[];destinations=[];reverses=[];na=[];ae=0;be=0
    for i in molecules:
        ast,aen=g['ap'][i:i+2];est,een=g['ep'][i:i+2]
        atoms.append(g['atom'][ast:aen]);bonds.append(g['bond'][est:een]);na.append(int(aen-ast))
        sources.append(g['src'][est:een]+ae);destinations.append(g['dst'][est:een]+ae);reverses.append(g['rev'][est:een]+be)
        ae+=int(aen-ast);be+=int(een-est)
    arrays=[np.concatenate(atoms),np.concatenate(bonds),np.concatenate(sources),np.concatenate(destinations),
            np.concatenate(reverses),np.repeat(np.arange(len(na),dtype=np.int64),na),np.array(na,np.float32)]
    return [torch.from_numpy(x).to('cuda') for x in arrays]

class TaskHeads(nn.Module):
    def __init__(self,dim,n):
        super().__init__();h=P['private_hidden']
        self.w=nn.Parameter(torch.empty(n,dim,h));self.bias=nn.Parameter(torch.zeros(n,h))
        self.out=nn.Parameter(torch.empty(n,h));self.ob=nn.Parameter(torch.zeros(n))
        nn.init.normal_(self.w,std=dim**-.5);nn.init.normal_(self.out,std=h**-.5)
        self.drop=nn.Dropout(P['dropout'])
    def forward(self,x):
        z=self.drop(nn.functional.silu(torch.einsum('bi,kih->bkh',x,self.w)+self.bias))
        return (z*self.out).sum(-1)+self.ob

class BondNet(nn.Module):
    def __init__(self,atomdim,bonddim,featdim,n):
        super().__init__();h=P['hidden'];dp=P['dropout']
        self.edge=nn.Linear(atomdim+bonddim,h);self.message=nn.Linear(h,h,bias=False)
        self.atom=nn.Linear(atomdim+h,h);self.drop=nn.Dropout(dp)
        self.pool=nn.Sequential(nn.Linear(2*h+1,h),nn.LayerNorm(h),nn.SiLU(),nn.Dropout(dp))
        self.feat=nn.Sequential(nn.Linear(featdim,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(.25),nn.Linear(256,h),nn.SiLU())
        self.graph_heads=TaskHeads(h,n);self.hybrid_heads=TaskHeads(2*h,n)
    def forward(self,g,x):
        atom,bond,src,dst,rev,mol,natoms=g
        initial=nn.functional.silu(self.edge(torch.cat([atom[src],bond],1)));h=initial
        for _ in range(P['depth']-1):
            incoming=h.new_zeros((len(atom),P['hidden']));incoming.index_add_(0,dst,h)
            h=self.drop(nn.functional.silu(initial+self.message(incoming[src]-h[rev])))
        incoming=h.new_zeros((len(atom),P['hidden']));incoming.index_add_(0,dst,h)
        atoms=nn.functional.silu(self.atom(torch.cat([atom,incoming],1)))
        summed=atoms.new_zeros((len(natoms),P['hidden']));summed.index_add_(0,mol,atoms)
        graph=self.pool(torch.cat([summed/natoms[:,None],summed/natoms[:,None].sqrt(),natoms.log1p()[:,None]],1))
        features=self.feat(x)
        return torch.stack([self.graph_heads(graph),self.hybrid_heads(torch.cat([graph,features],1))])

def fit(d,g,rows,query,seed,dest):
    assert os.name!='nt' and ROOT==Path(P['remote_root']) and torch.cuda.is_available()
    rows=np.asarray(rows,int);query=np.asarray(query,int);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    cache=dest/'predictions.npz';sig=signature()
    if cache.exists():
        z=np.load(cache);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],rows);np.testing.assert_array_equal(z['query'],query)
        return z['p'].copy()
    assert not set(rows)&set(query) and np.all(d['split'][rows]<2)
    if np.any(d['split'][query]<2):
        assert not set(d['rg'][rows])&set(d['rg'][query])
        assert not set(d['canonical'][d['mi'][rows]])&set(d['canonical'][d['mi'][query]])
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    n=len(d['tasks']);targets=[];indices=[];pw=[]
    y=np.zeros((len(d['smiles']),n),np.float32);mask=np.zeros_like(y,dtype=bool)
    for k in range(n):
        ix=rows[d['tid'][rows]==k];assert len(ix)
        tf=b.Target(d['y'][ix],d['metrics'][k]);targets.append(tf)
        mols=d['mi'][ix];unique=np.unique(mols);indices.append(unique)
        # No queried labels are ever placed into this label matrix.
        for mi in unique:
            values=d['y'][ix[mols==mi]]
            value=float(np.mean(values)) if d['metrics'][k] in ['AUROC','AUPRC'] else float(np.median(values))
            y[mi,k]=tf.encode(np.array([value]))[0];mask[mi,k]=True
        prevalence=float(np.mean(d['y'][ix])) if d['metrics'][k] in ['AUROC','AUPRC'] else 1.
        pw.append(min(3.,max(1.,np.sqrt((1-prevalence)/max(prevalence,.01)))))
    unique=np.unique(d['mi'][rows]);raw=np.concatenate([d['desc'],d['emb']],1)
    pre=b.Preprocessor().fit(raw[unique]);features=np.concatenate([d['fp2'],d['fcfp'],pre.transform(raw)],1)
    features=torch.tensor(features,device='cuda');labels=torch.tensor(y,device='cuda');observed=torch.tensor(mask,device='cuda')
    net=BondNet(g['atom'].shape[1],g['bond'].shape[1],features.shape[1],n).cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,P['steps'],eta_min=P['minimum_learning_rate'])
    t0=time.time();updates=np.zeros(n,int);snaps=[]
    def predict():
        net.eval();molecules=np.unique(d['mi'][query]);values=np.empty((2,len(d['smiles']),n),np.float32)
        with torch.inference_mode():
            for start in range(0,len(molecules),96):
                mm=molecules[start:start+96];values[:,mm]=net(pack(g,mm),features[mm]).cpu().numpy()
        out=np.empty((2,len(query)),float)
        for k in range(n):
            j=np.flatnonzero(d['tid'][query]==k);v=values[:,d['mi'][query[j]],k]
            if d['metrics'][k]=='Spearman':out[:,j]=v
            else:out[:,j]=targets[k].decode(v)
        assert np.isfinite(out).all();return out
    for step in range(1,P['steps']+1):
        task=int(rng.integers(n));mm=rng.choice(indices[task],size=min(P['batch_size'],len(indices[task])),replace=True)
        logits=net(pack(g,mm),features[mm]);by=labels[mm];bm=observed[mm];losses=[]
        for k in range(n):
            active=bm[:,k]
            if not active.any():continue
            updates[k]+=1;true=by[active,k];pred=logits[:,active,k];m=d['metrics'][k]
            if m in ['AUROC','AUPRC']:
                loss=nn.functional.binary_cross_entropy_with_logits(pred,true.expand_as(pred),pos_weight=pred.new_tensor(pw[k]))
                a=pred[:,true>.5];c=pred[:,true<.5]
                if a.shape[1] and c.shape[1]:loss=loss+.1*nn.functional.softplus(c[:,None]-a[:,:,None]).mean()
            elif m=='MAE':
                loss=.9*(pred-true).abs().mean()+.1*nn.functional.smooth_l1_loss(pred,true.expand_as(pred),beta=.3)
            else:
                loss=nn.functional.smooth_l1_loss(pred,true.expand_as(pred),beta=.3)
                diff=true[:,None]-true[None,:];active_pairs=diff.abs()>.15
                if active_pairs.any():loss=loss+.1*nn.functional.softplus(-diff[active_pairs].sign()*(pred[:,:,None]-pred[:,None,:])[:,active_pairs]).mean()
            losses.append(loss)
        loss=torch.stack(losses).mean();assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step();schedule.step()
        if step in P['snapshot_steps']:
            q=predict();snaps.append(q)
            checkpoint=dict(state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},target=targets,preprocessor=pre,
                            dimensions=[g['atom'].shape[1],g['bond'].shape[1],features.shape[1],n],seed=seed,step=step,signature=sig)
            path=dest/f'checkpoint_{step}.pt';torch.save(checkpoint,path)
            restored=torch.load(path,map_location='cpu',weights_only=False);net.load_state_dict(restored['state'])
            np.testing.assert_allclose(q,predict(),rtol=3e-5,atol=3e-5)
        if step%500==0 or step==P['steps']:
            b.emit('v27_fit',seed=seed,partition=str(dest.relative_to(OUT)),step=step,seconds=round(time.time()-t0),loss=float(loss.detach()))
        net.train()
    p=np.mean(snaps,0);np.savez_compressed(cache,p=p,fit=rows,query=query,signature=sig)
    b.save(dest/'metadata.json',dict(signature=sig,seed=seed,seconds=time.time()-t0,fit_rows=len(rows),query_rows=len(query),
       observed_task_updates=dict(zip(d['tasks'],updates.tolist())),unique_fit_molecules=len(unique),model_reload_passed=True,
       group_exclusion_checked=bool(np.any(d['split'][query]<2)),test_labels_used=False,local_gpu_training=False))
    del net,features,labels,observed;torch.cuda.empty_cache()
    return p

def selftest(d,g):
    # Check reverse bonds in the packed cache for every molecule.
    for i in range(len(d['smiles'])):
        es,ee=g['ep'][i:i+2];s=g['src'][es:ee];t=g['dst'][es:ee];r=g['rev'][es:ee]
        np.testing.assert_array_equal(s[r],t);np.testing.assert_array_equal(r[r],np.arange(len(r)))
    net=BondNet(g['atom'].shape[1],g['bond'].shape[1],10,len(d['tasks'])).cuda().eval()
    with torch.inference_mode():
        a=net(pack(g,[0,1,2]),torch.zeros((3,10),device='cuda')).cpu().numpy()
        c=net(pack(g,[2,0,1]),torch.zeros((3,10),device='cuda')).cpu().numpy()
        q=net(pack(g,[0]),torch.zeros((1,10),device='cuda')).cpu().numpy()
    np.testing.assert_allclose(a[:,[2,0,1]],c,rtol=3e-5,atol=3e-5)
    np.testing.assert_allclose(a[:,:1],q,rtol=3e-5,atol=3e-5)
    b.save(OUT/'selftest.json',dict(reverse_bonds_valid=True,query_batch_invariance=True,finite_features=True,test_labels_used=False))
    del net;torch.cuda.empty_cache()
