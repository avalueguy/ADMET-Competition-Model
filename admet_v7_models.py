"""Remote v7: complementary chemical views and supervised SMILES fine-tuning."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from sklearn.svm import SVC, SVR
from catboost import CatBoostClassifier, CatBoostRegressor
from rdkit import Chem, rdBase
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator, GetTopologicalTorsionGenerator
from admet_v6_features import load_task as load6
from admet_v6_models import Preprocess, Target, train_loss


def load_task(task):
    d=load6(task)
    schema={'v':7,'rdkit':rdBase.rdkitVersion,'views':['atom_pair_count2048_chiral','torsion_count2048_chiral','MACCS167']}
    key=hashlib.sha256(json.dumps([schema,d['smiles']],sort_keys=True).encode()).hexdigest()
    path=Path('admet_model_outputs_v7/features')/(task+'.npz')
    if path.exists():
        with np.load(path) as z:
            assert str(z['key'])==key
            d.update(ap=z['ap'],tt=z['tt'],maccs=z['maccs'])
    else:
        ag=GetAtomPairGenerator(fpSize=2048,includeChirality=True)
        tg=GetTopologicalTorsionGenerator(fpSize=2048,includeChirality=True)
        arrays=[np.zeros((len(d['smiles']),dim),np.float32) for dim in [2048,2048,167]]
        for i,s in enumerate(d['smiles']):
            m=Chem.MolFromSmiles(s)
            if m is None:continue
            arrays[0][i]=ag.GetCountFingerprintAsNumPy(m)
            arrays[1][i]=tg.GetCountFingerprintAsNumPy(m)
            arrays[2][i]=np.asarray(MACCSkeys.GenMACCSKeys(m))
        d.update(zip(['ap','tt','maccs'],arrays))
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.stem+'.'+str(os.getpid())+'.npz')
        np.savez_compressed(temporary,key=key,**dict(zip(['ap','tt','maccs'],arrays)))
        os.replace(temporary,path)
    d['v7_feature_key']=key
    return d


def fp_kernel(d,query,reference):
    """Generalized Tanimoto; no labels or fitted statistics enter fingerprints."""
    parts=[]
    for start in range(0,len(query),256):
        result=torch.zeros((len(query[start:start+256]),len(reference)),device='cuda')
        for name,w in [('count',.3),('ap',.25),('tt',.2),('maccs',.1)]:
            a=torch.as_tensor(d[name][query[start:start+256]],device='cuda')
            b=torch.as_tensor(d[name][reference],device='cuda')
            dot=a@b.T
            result+=w*dot/(a.square().sum(1)[:,None]+b.square().sum(1)[None,:]-dot).clamp_min(1e-8)
        parts.append(result.cpu().numpy())
    return np.concatenate(parts)


def full_kernel(d,query,reference,pre):
    k=fp_kernel(d,query,reference)
    a=pre.transform(d['desc'][query]);b=pre.transform(d['desc'][reference])
    for start in range(0,len(query),256):
        aa=torch.as_tensor(a[start:start+256],device='cuda');bb=torch.as_tensor(b,device='cuda')
        dist=(aa.square().sum(1)[:,None]+bb.square().sum(1)[None,:]-2*aa@bb.T).clamp_min(0)
        k[start:start+256]+=.15*torch.exp(-dist/a.shape[1]).cpu().numpy()
    return k.astype(np.float64)


def cat_x(d):
    return np.concatenate([np.log1p(d['count']),np.log1p(d['ap']),np.log1p(d['tt']),d['maccs'],d['desc'],d['emb']],1)


class FineTune(nn.Module):
    def __init__(self,dim):
        super().__init__()
        from transformers import AutoModel
        self.encoder=AutoModel.from_pretrained('admet_pretrained/chemberta_mlm',local_files_only=True,add_pooling_layer=False)
        self.desc=nn.Sequential(nn.Linear(dim,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.2))
        self.head=nn.Sequential(nn.Linear(864,192),nn.LayerNorm(192),nn.SiLU(),nn.Dropout(.25),nn.Linear(192,1))

    def forward(self,tokens,extra):
        h=self.encoder(**tokens).last_hidden_state
        mask=tokens['attention_mask'].unsqueeze(-1)
        mean=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.head(torch.cat([mean,h[:,0],self.desc(extra)],1)).squeeze(-1)


def prepare_tokens(d,protocol):
    if 'tokens' in d:return
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('admet_pretrained/chemberta_mlm',local_files_only=True)
    canonical=[];variants=[[],[]]
    for s in d['smiles']:
        mol=Chem.MolFromSmiles(s)
        cs=Chem.MolToSmiles(mol) if mol is not None else 'C'
        canonical.append(cs)
        for v in range(2):
            seed=int(hashlib.sha256((cs+str(v)).encode()).hexdigest()[:7],16)+1
            variants[v].append(Chem.MolToRandomSmilesVect(mol,1,randomSeed=seed)[0] if mol is not None else cs)
    # Augmented target strings contain no assay information. Holdouts use canonical only.
    d['tokens']=[tokenizer(texts,truncation=True,max_length=protocol['ft_max_tokens']) for texts in [canonical]+variants]
    d['tokenizer']=tokenizer
    d['truncated_canonical_count']=sum(len(v)>protocol['ft_max_tokens'] for v in tokenizer(canonical,truncation=False)['input_ids'])


def token_batch(d,ix,variant=0):
    values=[{k:d['tokens'][variant][k][i] for k in d['tokens'][variant]} for i in ix]
    return {k:v.cuda() for k,v in d['tokenizer'].pad(values,padding=True,return_tensors='pt').items()}


def fit(family,d,ix,y,metric,protocol):
    cls=metric in ['AUROC','AUPRC'];tf=None if cls else Target(y,metric)
    yy=y if cls else tf.encode(y)
    bundle=dict(family=family,classification=cls,target=tf,models=[],metric=metric,feature_key=d['v7_feature_key'])
    if family=='pharm_kernel':
        pre=Preprocess().fit(d['desc'][ix]);x=full_kernel(d,ix,ix,pre)
        model=(SVC(C=5,kernel='precomputed',class_weight='balanced',cache_size=1536) if cls else SVR(C=5,kernel='precomputed',epsilon=.05,cache_size=1536))
        model.fit(x,yy);bundle.update(models=[model],reference=np.asarray(ix),pre=pre)
        return bundle
    x=cat_x(d) if family=='pharm_cat' else d['desc']
    if family=='pharm_cat':
        # Fit-only unsupervised screening; exclude constant rare bits and reduce tree cost.
        nfp=2048*3+167
        columns=np.r_[np.argsort(np.var(x[ix,:nfp],axis=0),kind='stable')[-protocol['cat_fingerprint_columns']:],np.arange(nfp,x.shape[1])]
        bundle['columns']=columns;x=x[:,columns]
    pre=Preprocess().fit(x[ix]);xx=pre.transform(x[ix]);bundle['pre']=pre
    if family=='pharm_cat':
        for seed in protocol['seeds']:
            klass=CatBoostClassifier if cls else CatBoostRegressor
            model=klass(iterations=protocol['cat_iterations'],depth=6,learning_rate=.035,l2_leaf_reg=6,random_seed=seed,thread_count=6,
                loss_function='Logloss' if cls else ('MAE' if metric=='MAE' else 'RMSE'),verbose=False,allow_writing_files=False)
            model.fit(xx,yy);bundle['models'].append(model)
        return bundle
    assert family=='chemberta_ft'
    prepare_tokens(d,protocol);bundle.update(dim=xx.shape[1],protocol=protocol,truncated=d['truncated_canonical_count'])
    target=torch.as_tensor(yy,dtype=torch.float32,device='cuda');xx=torch.as_tensor(xx,device='cuda')
    pw=torch.tensor(min(3.,max(1.,np.sqrt((1-y.mean())/max(y.mean(),.01)))) if cls else 1.,device='cuda')
    for seed in protocol['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=FineTune(bundle['dim']).cuda()
        opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':2e-5},
                              {'params':list(model.desc.parameters())+list(model.head.parameters()),'lr':5e-4}],weight_decay=.01)
        epochs=protocol['ft_epochs'];sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=1e-6)
        for epoch in range(epochs):
            model.train();order=rng.permutation(len(ix))
            for start in range(0,len(ix),32):
                jj=order[start:start+32];tokens=token_batch(d,ix[jj],variant=epoch%3)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=model(tokens,xx[jj]);loss=train_loss(pred.float(),target[jj],metric,pw)
                if not torch.isfinite(loss):raise ValueError('nonfinite fine-tuning loss')
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            sched.step()
            if epoch+1 in [epochs-4,epochs]:bundle['models'].append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        del model
    return bundle


def predict(bundle,d,ix):
    fam=bundle['family'];cls=bundle['classification'];preds=[]
    assert bundle['feature_key']==d['v7_feature_key']
    if fam=='pharm_kernel':x=full_kernel(d,ix,bundle['reference'],bundle['pre'])
    else:
        raw=cat_x(d)[:,bundle['columns']] if fam=='pharm_cat' else d['desc']
        x=bundle['pre'].transform(raw[ix])
    if fam=='chemberta_ft':prepare_tokens(d,bundle['protocol'])
    for model in bundle['models']:
        if fam=='pharm_kernel':p=expit(model.decision_function(x)) if cls else model.predict(x)
        elif fam=='pharm_cat':p=model.predict_proba(x)[:,1] if cls else model.predict(x)
        else:
            net=FineTune(bundle['dim']).cuda();net.load_state_dict(model);net.eval();out=[]
            with torch.inference_mode():
                for start in range(0,len(ix),64):
                    out.append(net(token_batch(d,ix[start:start+64]),torch.as_tensor(x[start:start+64],device='cuda')).cpu().numpy())
            p=np.concatenate(out);p=expit(p) if cls else p;del net
        if not cls:p=bundle['target'].decode(p)
        preds.append(p)
    return np.mean(preds,axis=0)
