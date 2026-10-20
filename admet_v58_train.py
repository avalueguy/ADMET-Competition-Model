"""Remote supervised SMILES encoder tuning, identity augmentation and gated chemical fusion."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from transformers import AutoTokenizer,AutoModel
import admet_v24_models as b
import admet_v51_run as old

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v58'
P=json.loads((ROOT/'admet_v58_protocol.json').read_text(encoding='utf-8'))
MODEL=ROOT/'admet_pretrained/chemberta_mlm'
TOK=None

def signature():
    return hashlib.sha256((b.sha(__file__)+b.sha(ROOT/'admet_v58_protocol.json')+b.sha(MODEL/'pytorch_model.bin')).encode()).hexdigest()

def tokens(d):
    global TOK
    TOK=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    path=OUT/'tokens.pt';key=hashlib.sha256(json.dumps(d['smiles']).encode()).hexdigest()
    if path.exists():
        z=torch.load(path,weights_only=False);assert z['key']==key and z['max_tokens']==P['max_tokens'];return z['views']
    texts=[[],[],[]]
    for sm in d['smiles']:
        m=Chem.MolFromSmiles(sm)
        if m is None:
            for a in texts:a.append(sm)
            continue
        canonical=Chem.MolToSmiles(m,isomericSmiles=True);texts[0].append(canonical)
        seed=int(hashlib.sha256(canonical.encode()).hexdigest()[:7],16)+1
        rand=Chem.MolToRandomSmilesVect(m,2,randomSeed=seed,isomericSmiles=True)
        texts[1].append(rand[0]);texts[2].append(rand[1])
    views=[TOK(x,padding=False,truncation=True,max_length=P['max_tokens']) for x in texts]
    torch.save(dict(key=key,views=views,max_tokens=P['max_tokens']),path)
    return views

def batch(views,mol,variant):
    data=views[variant]
    raw=[{k:data[k][int(i)] for k in data} for i in mol]
    return {k:v.cuda() for k,v in TOK.pad(raw,padding=True,return_tensors='pt').items()}

class Net(nn.Module):
    def __init__(self,fpdim,densedim):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(MODEL,local_files_only=True,add_pooling_layer=False,attn_implementation='sdpa')
        dp=P['dropout']
        def layer(a,c):return nn.Sequential(nn.Linear(a,c),nn.LayerNorm(c),nn.SiLU(),nn.Dropout(dp))
        self.fp=layer(fpdim,192);self.dense=layer(densedim,96)
        self.struct=nn.Sequential(layer(288,192),nn.Linear(192,1))
        self.sequence=layer(768,192);self.gate=nn.Linear(480,192)
        self.hybrid=nn.Sequential(layer(480,192),nn.Linear(192,1))
    def forward(self,token,fp,dense):
        if self.training:fp=nn.functional.dropout(fp,.05,True)
        z=torch.cat([self.fp(fp),self.dense(dense)],1)
        h=self.encoder(**token).last_hidden_state
        mask=token['attention_mask'].bool()&(token['input_ids']!=TOK.bos_token_id)&(token['input_ids']!=TOK.eos_token_id)
        mean=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1).clamp_min(1)[:,None]
        seq=self.sequence(torch.cat([mean,h[:,0]],1));gate=self.gate(torch.cat([z,seq],1)).sigmoid()
        return torch.stack([self.struct(z).squeeze(-1),self.hybrid(torch.cat([z,seq*gate],1)).squeeze(-1)])

def fit(d,views,k,rows,query,dest):
    dest.mkdir(parents=True,exist_ok=True);sig=signature();cache=dest/'predictions.npz'
    if cache.exists():
        z=np.load(cache);assert str(z['signature'])==sig
        np.testing.assert_array_equal(rows,z['fit']);np.testing.assert_array_equal(query,z['query']);return z['p'].copy()
    assert not set(rows)&set(query) and np.all(d['split'][rows]<2)
    if np.any(d['split'][query]<2):assert not set(d['rg'][rows])&set(d['rg'][query])
    start=time.time();metric=d['metrics'][k];tf=b.Target(d['y'][rows],metric)
    mols=np.unique(d['mi'][np.r_[rows,query]]);lookup=np.full(len(d['smiles']),-1);lookup[mols]=np.arange(len(mols))
    rr=lookup[d['mi'][rows]];qq=lookup[d['mi'][query]]
    pre=b.Preprocessor().fit(np.concatenate([d['desc'],d['emb']],1)[np.unique(d['mi'][rows])])
    dense=pre.transform(np.concatenate([d['desc'][mols],d['emb'][mols]],1));fp=np.concatenate([d['fp2'][mols],d['fcfp'][mols]],1)
    xs=[torch.tensor(x,dtype=torch.float32,device='cuda') for x in [fp,dense]]
    target=torch.tensor(tf.encode(d['y'][rows]),device='cuda')
    _,inv,count=np.unique(d['canonical'][d['mi'][rows]],return_inverse=True,return_counts=True)
    weight=torch.tensor(1./count[inv],dtype=torch.float32,device='cuda')
    dims=[fp.shape[1],dense.shape[1]];parts=[]
    def predict(net):
        net.eval();outputs=[]
        with torch.inference_mode():
            for s in range(0,len(query),48):
                ii=qq[s:s+48];mm=d['mi'][query[s:s+48]]
                with torch.autocast('cuda',dtype=torch.bfloat16):v=net(batch(views,mm,0),*[x[ii] for x in xs])
                outputs.append(v.float().cpu().numpy())
        raw=np.concatenate(outputs,axis=1)
        return tf.decode(raw) if metric=='MAE' else raw
    for seed,enc_lr in zip(P['seeds'],P['encoder_learning_rates']):
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        net=Net(*dims).cuda();enc=list(net.encoder.parameters());ids={id(x) for x in enc}
        head=[x for x in net.parameters() if id(x) not in ids]
        opt=torch.optim.AdamW([{'params':enc,'lr':enc_lr},{'params':head,'lr':P['head_learning_rate']}],weight_decay=.018)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,P['steps'],eta_min=1e-6);snap=[]
        for step in range(1,P['steps']+1):
            jj=rng.choice(len(rows),size=min(P['batch_size'],len(rows)),replace=False);ii=rr[jj];variant=step%3
            net.train()
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(batch(views,d['mi'][rows[jj]],variant),*[x[ii] for x in xs])
            pred=pred.float();y=target[jj];w=weight[jj]
            if metric in ['AUROC','AUPRC']:
                loss=nn.functional.binary_cross_entropy_with_logits(pred,y.expand_as(pred),reduction='none')
                loss=(loss*w).mean();pos=pred[:,y>.5];neg=pred[:,y<.5]
                if pos.shape[1] and neg.shape[1]:loss+=.12*nn.functional.softplus(neg[:,None,:]-pos[:,:,None]).mean()
            elif metric=='MAE':loss=((pred-y).abs()*w).mean()
            else:
                loss=nn.functional.smooth_l1_loss(pred,y.expand_as(pred),beta=.4)
                diff=y[:,None]-y[None,:];active=diff.abs()>.15
                if active.any():loss+=.2*nn.functional.softplus(-diff[active].sign()*(pred[:,:,None]-pred[:,None,:])[:,active]).mean()
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();sched.step()
            if step in P['snapshots']:
                p=predict(net);snap.append(p)
                torch.save(dict(state={n:v.cpu() for n,v in net.state_dict().items()},dimensions=dims,pre=pre,target=tf,signature=sig),dest/f'{seed}_{step}.pt')
            if step%60==0:b.emit('v58_fit',task=d['tasks'][k],partition=dest.name,seed=seed,step=step,seconds=round(time.time()-start))
        parts.append(np.mean(snap,0));del net,opt;torch.cuda.empty_cache()
    p=np.mean(parts,0);assert np.isfinite(p).all()
    np.savez_compressed(cache,p=p,fit=rows,query=query,signature=sig)
    b.save(dest/'metadata.json',dict(signature=sig,fit_rows=len(rows),query_rows=len(query),
        test_labels_used=False,preprocessor_fit_only=True,encoder_weights_updated=True,
        random_smiles_equivalence=True,query_augmentation=False,local_gpu_training=False,seconds=time.time()-start))
    return p

def variants(p,metric):
    a=p if metric=='MAE' else np.stack([b.rank(x) for x in p])
    return np.concatenate([p,np.mean(a,0)[None]])

def main():
    b.remote_only();OUT.mkdir(exist_ok=True);d=b.build_data();views=tokens(d);basev,basec=old.baseline(d);folds=np.load(b.OUT/'folds.npy');records=[]
    np.save(OUT/'baseline_valid.npy',basev);np.save(OUT/'baseline_cv_proxy.npy',basec)
    for task in P['tasks']:
        k=d['tasks'].index(task);metric=d['metrics'][k];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        cp=np.full((3,len(d['y'])),np.nan)
        for f in range(3):
            rr=tr[folds[tr]!=f];q=tr[folds[tr]==f]
            cp[:,q]=variants(fit(d,views,k,rr,q,OUT/task/f'fold{f}'),metric)
        np.save(OUT/task/'cv_predictions.npy',cp)
        vp=variants(fit(d,views,k,tr,vi,OUT/task/'valid'),metric);grid=[]
        for v in range(3):
            for w in P['weights']:
                gains=[b.delta(metric,d['y'][ix],basec[ix],b.blend(metric,basec[ix],cp[v,ix],w)) for f in range(3) for ix in [tr[folds[tr]==f]]]
                vg=b.delta(metric,d['y'][vi],basev[vi],b.blend(metric,basev[vi],vp[v],w))
                obj=float(.85*np.mean(gains)+.15*vg-.07*np.std(gains))
                grid.append(dict(variant=v,weight=w,cv_gains=gains,cv_mean=float(np.mean(gains)),valid_gain=vg,objective=obj))
        held=[max(grid,key=lambda x:np.mean([g for j,g in enumerate(x['cv_gains']) if j!=f]))['cv_gains'][f] for f in range(3)]
        good=[x for x in grid if x['cv_mean']>.001 and x['objective']>.001 and sum(g>0 for g in x['cv_gains'])>=2]
        selected=max(good,key=lambda x:x['objective']) if good and np.mean(held)>0 else None
        r=dict(task=task,metric=metric,selection=selected,grid=grid,held_gains=held,held_mean=float(np.mean(held)))
        records.append(r);b.save(OUT/'analysis.json',dict(status='IN_PROGRESS',signature=signature(),tasks=records));b.emit('v58_selection',**{k:v for k,v in r.items() if k!='grid'})
    b.save(OUT/'analysis.json',dict(status='FROZEN',signature=signature(),tasks=records,official_score=None))

if __name__=='__main__':main()
