"""Remote-only, versioned v24 features and multi-task models."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit
from scipy.stats import rankdata,norm,spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score,mean_absolute_error
from rdkit import Chem,RDLogger,rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator,GetMorganFeatureAtomInvGen

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_model_outputs_v24'
P=json.loads((ROOT/'admet_v24_protocol.json').read_text())
RDLogger.DisableLog('rdApp.*')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)
def emit(stage,**kw):print(json.dumps(dict(stage=stage,**kw),ensure_ascii=False),flush=True)
def remote_only():
    assert os.name!='nt' and ROOT==Path(P['remote_root']), 'Remote-only training'
    assert torch.cuda.is_available()
    torch.set_num_threads(3);torch.set_num_interop_threads(2)
    torch.cuda.set_per_process_memory_fraction(.40)
def signature():
    files=[ROOT/'admet_v24_protocol.json',ROOT/'admet_v24_models.py',ROOT/'admet_v24_run.py']
    files+=sorted((ROOT/'admet_competition_data/data_benchmark').glob('*/*.csv'))
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},sort_keys=True).encode()).hexdigest()
def metric(m,y,p):
    if len(y)==0:return np.nan
    if m=='MAE':return float(mean_absolute_error(y,p))
    if m=='Spearman':
        r=float(spearmanr(y,p).statistic);return r if np.isfinite(r) else 0.
    if len(np.unique(y))<2:return np.nan
    return float((roc_auc_score if m=='AUROC' else average_precision_score)(y,p))
def delta(m,y,a,b):
    x,z=metric(m,y,a),metric(m,y,b)
    return (x-z)/max(x,1e-8) if m=='MAE' else z-x
def rank(x):return (rankdata(x,method='average')-.5)/len(x)
def blend(m,a,b,w):
    if w==0:return np.array(a,copy=True)
    if m!='MAE':a,b=rank(a),rank(b)
    return (1-w)*a+w*b

def build_data():
    template=pd.read_csv(ROOT/'admet_official/submission_template.csv',dtype=str,keep_default_na=False)
    tasks=template.task.drop_duplicates().tolist();metrics=[template.loc[template.task.eq(t),'metric'].iloc[0] for t in tasks]
    frames={};mols={};slist=[];mi=[];tid=[];split=[];labels=[];row=[]
    for k,t in enumerate(tasks):
        for si,s in enumerate(['train','valid','test']):
            df=pd.read_csv(ROOT/'admet_competition_data/data_benchmark'/t/(s+'.csv'),keep_default_na=False)
            frames[t,s]=df
            if s=='test':
                tx=template.loc[template.task.eq(t)]
                assert tx.smiles.tolist()==df.Drug.tolist()
                np.testing.assert_array_equal(tx.sample_idx.astype(int),np.arange(len(df)))
            for j,sm in enumerate(df.Drug.astype(str)):
                if sm not in mols:mols[sm]=len(slist);slist.append(sm)
                mi.append(mols[sm]);tid.append(k);split.append(si);row.append(j)
                labels.append(float(df.Y.iloc[j]) if s!='test' else np.nan)
    key=hashlib.sha256(json.dumps(slist).encode()).hexdigest()
    dest=OUT/'features';dest.mkdir(parents=True,exist_ok=True)
    fpfile=dest/'chemistry.npz';meta=dest/'chemistry.json'
    schema=dict(smiles_sha=key,rdkit=rdBase.rdkitVersion,r2=2048,fcfp3=2048,source_sha=sha(__file__))
    if fpfile.exists():
        assert json.loads(meta.read_text())==schema
        z=np.load(fpfile);fp2=z['fp2'];fcfp=z['fcfp'];desc=z['desc'];groups=z['groups'];canonical=z['canonical']
    else:
        fns=[(n,f) for n,f in Descriptors.descList if n not in {'Ipc','AvgIpc','BalabanJ'}]
        fp2=np.zeros((len(slist),2048),np.float32);fcfp=np.zeros_like(fp2);desc=np.full((len(slist),len(fns)),np.nan,np.float32)
        groups=[];canonical=[]
        gens=[GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True),GetMorganGenerator(radius=3,fpSize=2048,includeChirality=True,atomInvariantsGenerator=GetMorganFeatureAtomInvGen())]
        invalid=0
        for i,sm in enumerate(slist):
            mol=Chem.MolFromSmiles(sm)
            if mol is None:invalid+=1;canonical.append('INVALID:'+sm);groups.append('INVALID:'+sm);continue
            can=Chem.MolToSmiles(mol,isomericSmiles=False);canonical.append(can)
            sc=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False);groups.append(sc or 'ACYCLIC:'+can)
            fp2[i]=gens[0].GetCountFingerprintAsNumPy(mol);fcfp[i]=gens[1].GetCountFingerprintAsNumPy(mol)
            for j,(_,fn) in enumerate(fns):
                try:v=float(fn(mol))
                except Exception:continue
                if np.isfinite(v) and abs(v)<1e10:desc[i,j]=v
            if i%10000==0:emit('chem_features',done=i,total=len(slist))
        groups=np.array(groups);canonical=np.array(canonical)
        # Merge any scaffold keys connected through a nonisomeric molecular identity.
        parent={g:g for g in groups}
        def find(x):
            while x!=parent[x]:parent[x]=parent[parent[x]];x=parent[x]
            return x
        first={}
        for c,g in zip(canonical,groups):
            if c in first:parent[find(g)]=find(first[c])
            else:first[c]=g
        groups=np.array([find(g) for g in groups])
        np.savez_compressed(fpfile,fp2=fp2,fcfp=fcfp,desc=desc,groups=groups,canonical=canonical);save(meta,schema)
        emit('chem_features_done',unique_smiles=len(slist),invalid=invalid)
    efile=dest/'chemberta.npy';emeta=dest/'chemberta.json'
    weights=ROOT/'admet_pretrained/chemberta_mlm'
    eschema=dict(smiles_sha=key,weights_sha=sha(weights/'pytorch_model.bin'),pool='masked_mean_no_special',max_length=256)
    if efile.exists():assert json.loads(emeta.read_text())==eschema;emb=np.load(efile)
    else:
        from transformers import AutoTokenizer,AutoModel
        # Feature initialization must not consume per-fit RNG state.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(24)
            tok=AutoTokenizer.from_pretrained(weights,local_files_only=True)
            model=AutoModel.from_pretrained(weights,local_files_only=True,add_pooling_layer=False).cuda().eval()
            parts=[]
            with torch.inference_mode():
                for start in range(0,len(slist),96):
                    q=tok(slist[start:start+96],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
                    h=model(**q).last_hidden_state
                    mask=q.attention_mask.bool() & (q.input_ids!=tok.bos_token_id) & (q.input_ids!=tok.eos_token_id)
                    v=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
                    parts.append(v.cpu().numpy())
                    if start%9600==0:emit('embedding',done=start,total=len(slist))
            emb=np.concatenate(parts);del model
        np.save(efile,emb);save(emeta,eschema);torch.cuda.empty_cache()
    # Reuse exact v23 label-free feature cache to make the comparator reproducible.
    raw_old=[]
    for t in tasks:
        for s in ['train','valid','test']:
            z=np.load(ROOT/'admet_model_outputs_v23/features'/f'{t}_{s}.npz');raw_old.append(z['x'])
    d=dict(tasks=tasks,metrics=metrics,frames=frames,smiles=slist,mi=np.array(mi),tid=np.array(tid),split=np.array(split),y=np.array(labels),row=np.array(row),
           fp2=np.log1p(fp2).clip(0,6),fcfp=np.log1p(fcfp).clip(0,6),desc=desc,emb=emb,groups=groups,canonical=canonical,old=np.concatenate(raw_old))
    assert not np.isfinite(d['y'][d['split']==2]).any()
    d['rg']=groups[d['mi']]
    return d

def folds(d):
    train=np.flatnonzero(d['split']==0);groups,inv=np.unique(d['rg'][train],return_inverse=True)
    counts=np.zeros((len(groups),len(d['tasks'])),int)
    np.add.at(counts,(inv,d['tid'][train]),1)
    rel=counts/np.maximum(counts.sum(0),1);rng=np.random.default_rng(P['fold_seed'])
    order=np.lexsort((rng.random(len(groups)),-np.max(rel,1)))
    mass=np.zeros((P['folds'],len(d['tasks'])));assigned=np.empty(len(groups),int)
    for g in order:
        costs=[]
        for f in range(P['folds']):
            tmp=mass.copy();tmp[f]+=rel[g];costs.append(np.square(tmp).sum())
        f=int(np.argmin(np.array(costs)+rng.random(P['folds'])*1e-12));mass[f]+=rel[g];assigned[g]=f
    out=np.full(len(d['y']),-1);out[train]=assigned[inv]
    for f in range(P['folds']):
        a=train[out[train]!=f];b=train[out[train]==f]
        assert not set(d['rg'][a])&set(d['rg'][b])
        assert not set(d['canonical'][d['mi'][a]])&set(d['canonical'][d['mi'][b]])
    save(OUT/'split_audit.json',dict(seed=P['fold_seed'],fold_task_fractions=mass.tolist(),global_scaffold_overlap=0,nonisomeric_overlap=0,
              note='New group folds over historically reused training labels; not new data'))
    np.save(OUT/'folds.npy',out)
    return out

class Preprocessor:
    def fit(self,x):
        self.med=np.nanmedian(x,axis=0);self.med=np.nan_to_num(self.med)
        z=np.where(np.isfinite(x),x,self.med);self.mu=z.mean(0);self.sd=z.std(0).clip(1e-4)
        return self
    def transform(self,x):return np.clip((np.where(np.isfinite(x),x,self.med)-self.mu)/self.sd,-8,8).astype(np.float32)
class Target:
    def __init__(self,y,m,legacy=False):
        self.m=m;self.legacy=legacy
        self.center=float(np.median(y));self.scale=max(float(np.quantile(y,.75)-np.quantile(y,.25)),.001)
        if m=='Spearman' and not legacy:
            self.knots,idx=np.unique(y,return_index=True);self.z=norm.ppf(np.clip((rankdata(y)-.5)/len(y),1e-5,1-1e-5))[idx]
    def encode(self,y):
        if self.m in ['AUROC','AUPRC']:return y.astype(np.float32)
        if self.m=='Spearman' and not self.legacy:return np.interp(y,self.knots,self.z).astype(np.float32)
        return ((y-self.center)/self.scale).astype(np.float32)
    def decode(self,q):
        if self.m in ['AUROC','AUPRC']:return expit(q)
        if self.m=='Spearman' and not self.legacy:return np.interp(q,self.z,self.knots)
        return q*self.scale+self.center

class LegacyNet(nn.Module):
    def __init__(self,n):
        super().__init__();self.back=nn.Sequential(nn.Linear(1040,384),nn.LayerNorm(384),nn.SiLU(),nn.Dropout(.12),nn.Linear(384,192),nn.LayerNorm(192),nn.SiLU(),nn.Dropout(.10))
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(192,64),nn.SiLU(),nn.Dropout(.10),nn.Linear(64,1)) for _ in range(n)])
    def forward(self,x,k):return self.heads[k](self.back(x[0])).squeeze(-1)
class GatedNet(nn.Module):
    def __init__(self,dims,n):
        super().__init__();h=P['hidden'];dp=P['dropout']
        self.enc=nn.ModuleList([nn.Sequential(nn.Linear(dim,h),nn.LayerNorm(h),nn.SiLU(),nn.Dropout(dp)) for dim in dims])
        self.task=nn.Embedding(n,24)
        self.view_gate=nn.Sequential(nn.Linear(h+24,64),nn.SiLU(),nn.Linear(64,len(dims)))
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(h,h),nn.LayerNorm(h),nn.SiLU(),nn.Dropout(dp),nn.Linear(h,h)) for _ in range(3)])
        self.expert_gate=nn.Linear(h+24,3)
        self.private=nn.ModuleList([nn.Sequential(nn.Linear(h,64),nn.SiLU(),nn.Dropout(dp)) for _ in range(n)])
        self.head=nn.ModuleList([nn.Linear(h+64,1) for _ in range(n)])
    def forward(self,x,k):
        vv=torch.stack([enc(a) for enc,a in zip(self.enc,x)],1);t=self.task.weight[k].expand(len(vv),-1)
        w=self.view_gate(torch.cat([vv.mean(1),t],1)).softmax(1)
        h=(vv*w.unsqueeze(-1)).sum(1);gate=self.expert_gate(torch.cat([h,t],1)).softmax(1)
        shared=h+(torch.stack([m(h) for m in self.experts],1)*gate.unsqueeze(-1)).sum(1)
        return self.head[k](torch.cat([shared,self.private[k](h)],1)).squeeze(-1)

def train_model(d,fit,query,family,seed,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    cache=dest/'predictions.npz';sig=signature()
    if cache.exists():
        z=np.load(cache);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],query)
        return z['p'].copy()
    assert not set(fit)&set(query);assert np.all(d['split'][fit]!=2)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    legacy=family=='legacy';n=len(d['tasks']);targets=[];train_ix=[];yy=[];weights=[];posweights=[]
    for k in range(n):
        ids=fit[d['tid'][fit]==k];assert len(ids)>0
        tf=Target(d['y'][ids],d['metrics'][k],legacy);targets.append(tf);train_ix.append(ids)
        yy.append(torch.tensor(tf.encode(d['y'][ids]),device='cuda'))
        can=d['canonical'][d['mi'][ids]];_,iv,c=np.unique(can,return_inverse=True,return_counts=True)
        weights.append(torch.tensor(1/np.sqrt(c[iv]),dtype=torch.float32,device='cuda'))
        prevalence=float(np.mean(d['y'][ids])) if d['metrics'][k] in ['AUROC','AUPRC'] else 1.
        posweights.append(min(4.,max(1.,np.sqrt((1-prevalence)/max(prevalence,.01)))))
    pre=None;oldpre=None
    if legacy:
        # Exact v23 preprocessing recipe: per-task and per-query descriptor normalization.
        import admet_v23_remote_mtl as old
        whole=np.zeros_like(d['old']);oldpre=[]
        for k,ix in enumerate(train_ix):
            iq=query[d['tid'][query]==k]
            whole[ix]=old.prep(d['old'][ix]);
            if len(iq):whole[iq]=old.prep(d['old'][iq])
        xs=[torch.tensor(whole,device='cuda')];lookup=np.arange(len(d['y']));net=LegacyNet(n).cuda()
    else:
        unique=np.unique(d['mi'][fit]);pre=Preprocessor().fit(np.concatenate([d['desc'][unique],d['emb'][unique]],1))
        dense=pre.transform(np.concatenate([d['desc'],d['emb']],1))
        xs=[torch.tensor(x,device='cuda') for x in [d['fp2'],d['fcfp'],dense]]
        lookup=d['mi'];net=GatedNet([x.shape[1] for x in xs],n).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=7e-4 if legacy else P['learning_rate'],weight_decay=.012 if legacy else P['weight_decay'])
    steps=1600 if legacy else P['steps'];snaps=[1600] if legacy else P['snapshot_steps']
    schedule=None if legacy else torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=4e-5)
    preds=[];t0=time.time()
    def predict():
        net.eval();result=np.full(len(query),np.nan)
        with torch.inference_mode():
            for k in range(n):
                jj=np.flatnonzero(d['tid'][query]==k)
                if not len(jj):continue
                q=[]
                for start in range(0,len(jj),512):
                    active=jj[start:start+512];rr=lookup[query[active]]
                    q.append(net([x[rr] for x in xs],k).cpu().numpy())
                result[jj]=targets[k].decode(np.concatenate(q))
        assert np.isfinite(result).all();return result
    for step in range(1,steps+1):
        net.train();k=int(rng.integers(n));ids=train_ix[k];m=d['metrics'][k]
        ii=rng.integers(len(ids),size=min(512 if legacy else P['batch_size'],len(ids)))
        bx=[x[lookup[ids[ii]]] for x in xs];by=yy[k][ii];p=net(bx,k)
        if m in ['AUROC','AUPRC']:
            pw=min(4.,max(1.,np.sqrt((1-float(by.mean()))/max(float(by.mean()),.01)))) if legacy else posweights[k]
            losses=nn.functional.binary_cross_entropy_with_logits(p,by,pos_weight=p.new_tensor(pw),reduction='none')
            pair=0.
            if not legacy:
                a=p[by>.5];b=p[by<.5]
                if len(a) and len(b):pair=.1*nn.functional.softplus(b[None]-a[:,None]).mean()
        elif legacy:losses=nn.functional.smooth_l1_loss(p,by,beta=.25 if m=='MAE' else .35,reduction='none');pair=0.
        elif m=='MAE':losses=.7*(p-by).abs()+.3*nn.functional.smooth_l1_loss(p,by,beta=.25,reduction='none');pair=0.
        else:
            losses=nn.functional.smooth_l1_loss(p,by,beta=.3,reduction='none')
            diff=by[:,None]-by[None,:];mask=diff.abs()>.15
            pair=.15*nn.functional.softplus(-diff[mask].sign()*(p[:,None]-p[None,:])[mask]).mean() if mask.any() else 0.
        loss=losses.mean() if legacy else (losses*weights[k][ii]).sum()/weights[k][ii].sum()+pair
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
        if schedule:schedule.step()
        if step in snaps:
            q=predict();preds.append(q)
            checkpoint=dict(state={a:b.cpu() for a,b in net.state_dict().items()},target=targets,preprocessor=pre,family=family,seed=seed,step=step,signature=sig,
                            dimensions=[x.shape[1] for x in xs])
            torch.save(checkpoint,dest/f'checkpoint_{step}.pt')
            reloaded=torch.load(dest/f'checkpoint_{step}.pt',weights_only=False,map_location='cpu')
            net.load_state_dict(reloaded['state']);np.testing.assert_allclose(predict(),q,rtol=1e-5,atol=1e-5)
        if step%800==0 or step==steps:emit('fit',family=family,seed=seed,partition=str(dest.relative_to(OUT)),step=step,seconds=round(time.time()-t0),loss=float(loss.detach()))
    prediction=np.mean(preds,axis=0)
    np.savez_compressed(cache,fit=fit,query=query,p=prediction,signature=sig)
    save(dest/'metadata.json',dict(signature=sig,seed=seed,family=family,fit_rows=len(fit),query_rows=len(query),snapshots=snaps,
          model_reload_passed=True,test_labels_used=False,seconds=time.time()-t0))
    del net,xs,yy;torch.cuda.empty_cache()
    return prediction
