"""Frozen self-supervised MoLFormer representations with kernel and neural heads."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from sklearn.svm import SVC,SVR
from admet_v6_models import Preprocess,Target,train_loss
from admet_v8_models import RobustTarget
from admet_v11_models import load_task as load11,references,kernel as sparse_kernel
from admet_v12_models import inner_split


def load_task(task):
    d=load11(task)
    with np.load(Path('admet_pretrained/molformer_embeddings')/(task+'.npz')) as z:
        assert str(z['digest'])==hashlib.sha256(json.dumps(d['smiles']).encode()).hexdigest()
        assert str(z['revision'])=='7b12d946c181a37f6012b9dc3b002275de070314'
        d['molformer']=z['x'].copy()
    d['v13_feature_key']=hashlib.sha256(('v13|'+d['v11_feature_key']+'|7b12d946c181a37f6012b9dc3b002275de070314').encode()).hexdigest()
    return d


def target(y,metric):
    return None if metric in ['AUROC','AUPRC'] else (RobustTarget(y,metric) if metric=='MAE' else Target(y,metric))


def kernel(d,ix,bundle):
    out=np.zeros((len(ix),len(bundle['reference'])),float)
    for name,pre,ref,weight in bundle['rbf_views']:
        x=pre.transform(d[name][ix])
        for start in range(0,len(ix),256):
            a=x[start:start+256].astype(np.float64);b=ref.astype(np.float64)
            dist=np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0)/a.shape[1]
            out[start:start+256]+=weight*(np.exp(-dist/.5)+np.exp(-dist/2))/2
    if bundle['family']=='mol_fusion':out=.6*out+.4*sparse_kernel(d,ix,bundle['sparse'])
    return out


class NeuralHead(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(dim,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.3))
        self.residual=nn.Sequential(nn.Linear(256,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.3),nn.Linear(256,256))
        self.head=nn.Sequential(nn.GELU(),nn.Dropout(.2),nn.Linear(256,1))
    def forward(self,x):
        h=self.project(x);return self.head(h+self.residual(h)).squeeze(-1)


def raw_x(d):return np.concatenate([d['molformer'],d['desc'],d['maccs']],axis=1)


def fit(family,d,ix,y,metric,p):
    from admet_v6_runner import remote_only,metrics,kind,utility
    remote_only();ix=np.asarray(ix,int);y=np.asarray(y,float);cls=metric in ['AUROC','AUPRC'];tf=target(y,metric)
    bundle=dict(family=family,feature_key=d['v13_feature_key'],metric=metric,classification=cls,target=tf,reference=ix,models=[])
    if family in ['mol_kernel','mol_fusion']:
        bundle['rbf_views']=[]
        for name,weight in [('molformer',.9),('desc',.1)]:
            pre=Preprocess().fit(d[name][ix]);bundle['rbf_views'].append((name,pre,pre.transform(d[name][ix]),weight))
        if family=='mol_fusion':
            pre=Preprocess().fit(d['desc'][ix]);bundle['sparse']=dict(pre=pre,reference_desc=pre.transform(d['desc'][ix]),references=references(d,ix))
        k=kernel(d,ix,bundle);np.testing.assert_allclose(k,k.T,atol=1e-7)
        for c in p['kernel_cs']:
            model=SVC(C=c,kernel='precomputed',class_weight='balanced',cache_size=1024) if cls else SVR(C=c,kernel='precomputed',epsilon=.03,cache_size=1024)
            model.fit(k,y if cls else tf.encode(y));bundle['models'].append(model)
        return bundle
    assert family=='mol_mlp'
    a,b,splitseed=inner_split(d,ix,y,metric);x=raw_x(d)
    pre=Preprocess().fit(x[ix[a]]);z=pre.transform(x[ix]);inner_tf=target(y[a],metric)
    yy=y if cls else inner_tf.encode(y)
    xx=torch.tensor(z,device='cuda');labels=torch.tensor(yy,dtype=torch.float32,device='cuda')
    chosen=[];curve=[]
    for seed in p['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);net=NeuralHead(z.shape[1]).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.03)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,p['mlp_epochs'],eta_min=2e-5)
        pw=torch.tensor(min(3.,max(1.,np.sqrt((1-y[a].mean())/max(y[a].mean(),.01)))) if cls else 1.,device='cuda')
        best=(-float('inf'),10)
        for epoch in range(1,p['mlp_epochs']+1):
            net.train();order=rng.permutation(a)
            for start in range(0,len(a),64):
                j=order[start:start+64];loss=train_loss(net(xx[j]),labels[j],metric,pw)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
            sched.step()
            if epoch%10==0:
                net.eval()
                with torch.inference_mode():q=net(xx[b]).cpu().numpy()
                q=expit(q) if cls else inner_tf.decode(q);u=utility(metric,metrics(kind(metric),y[b],q));curve.append(dict(seed=seed,epoch=epoch,utility=float(u)))
                if u>best[0]:best=(u,epoch)
        chosen.append(best[1]);del net
    pre=Preprocess().fit(x[ix]);xx=torch.tensor(pre.transform(x[ix]),device='cuda');labels=torch.tensor(y if cls else tf.encode(y),dtype=torch.float32,device='cuda')
    pw=torch.tensor(min(3.,max(1.,np.sqrt((1-y.mean())/max(y.mean(),.01)))) if cls else 1.,device='cuda')
    for seed,epochs in zip(p['seeds'],chosen):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);net=NeuralHead(x.shape[1]).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.03)
        # Same full-length schedule as inner fit, stopped at the chosen epoch.
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,p['mlp_epochs'],eta_min=2e-5)
        for epoch in range(epochs):
            net.train();order=rng.permutation(len(ix))
            for start in range(0,len(ix),64):
                j=order[start:start+64];loss=train_loss(net(xx[j]),labels[j],metric,pw)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
            sched.step()
        bundle['models'].append({k:v.detach().cpu().clone() for k,v in net.state_dict().items()});del net
    bundle.update(pre=pre,dim=x.shape[1],epochs=chosen,inner_seed=splitseed,inner_fit_rows=ix[a],inner_hold_rows=ix[b],inner_curve=curve)
    return bundle


def predict(bundle,d,ix):
    assert bundle['feature_key']==d['v13_feature_key'];values=[]
    if bundle['family'] in ['mol_kernel','mol_fusion']:
        k=kernel(d,ix,bundle)
        for m in bundle['models']:values.append(m.decision_function(k) if bundle['classification'] else m.predict(k))
    else:
        x=bundle['pre'].transform(raw_x(d)[ix]);net=NeuralHead(bundle['dim']).cuda()
        for state in bundle['models']:
            net.load_state_dict(state);net.eval();parts=[]
            with torch.inference_mode():
                for start in range(0,len(ix),128):parts.append(net(torch.tensor(x[start:start+128],device='cuda')).cpu().numpy())
            values.append(np.concatenate(parts))
        del net
    p=np.mean(values,axis=0)
    return expit(p) if bundle['classification'] else bundle['target'].decode(p)
