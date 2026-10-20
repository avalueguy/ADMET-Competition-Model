"""Remote multiview attention and embedding boosting, with fit-only model selection."""
import hashlib,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from scipy.special import expit
from catboost import CatBoostClassifier,CatBoostRegressor
from admet_v15_models import load_task,fit as historical_fit,predict as historical_predict
from admet_v6_models import Preprocess,train_loss
from admet_v8_models import RobustTarget
from admet_v12_models import inner_split

def recipe():return json.loads(Path('admet_v16_protocol.json').read_text())
def transform(y,metric):return None if metric in ['AUROC','AUPRC'] else RobustTarget(y,metric)
def decode(q,tf):return expit(q) if tf is None else tf.decode(q)

def views(d,ix):
    return [np.concatenate([d['molformer'][ix],d['emb'][ix]],1),
        np.log1p(d['count'][ix]),np.log1p(d['fcfp'][ix]),
        np.concatenate([d['desc'][ix],d['maccs'][ix]],1)]

class ViewAttention(nn.Module):
    def __init__(self,dims,hidden,dropout):
        super().__init__();self.dropout=dropout
        self.encoders=nn.ModuleList([nn.Sequential(nn.Linear(k,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(.2)) for k in dims])
        self.gate=nn.Sequential(nn.Linear(hidden*len(dims),hidden),nn.GELU(),nn.Linear(hidden,len(dims)))
        self.residual=nn.Sequential(nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(.25),nn.Linear(hidden,hidden))
        self.head=nn.Sequential(nn.GELU(),nn.Dropout(.2),nn.Linear(hidden,1))
    def forward(self,x):
        h=torch.stack([m(a) for m,a in zip(self.encoders,x)],1);g=self.gate(h.flatten(1))
        if self.training and self.dropout:
            mask=torch.rand_like(g)<self.dropout
            # Always retain at least one view; the rescue view is sampled uniformly.
            rescue=torch.randint(g.shape[1],(len(g),),device=g.device)
            mask[torch.arange(len(g),device=g.device),rescue]=False;g=g.masked_fill(mask,-1e4)
        z=(h*g.softmax(1).unsqueeze(-1)).sum(1)
        return self.head(z+self.residual(z)).squeeze(-1)

def neural_fit(d,ix,y,metric,p):
    from admet_v6_runner import metrics,kind,utility
    a,b,splitseed=inner_split(d,ix,y,metric);raw=views(d,ix);dims=[x.shape[1] for x in raw]
    epochs=[];curves=[];states=[]
    def tensors(pre,tf,active_raw,active_y):
        return [torch.tensor(pp.transform(x),device='cuda') for pp,x in zip(pre,active_raw)],torch.tensor(active_y if tf is None else tf.encode(active_y),dtype=torch.float32,device='cuda')
    def train_one(seed,pre,tf,fitrows,nepochs,hold=None):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);net=ViewAttention(dims,p['hidden'],p['view_dropout']).cuda()
        fit_raw=[x[fitrows] for x in raw];fit_y=y[fitrows]
        xx,yy=tensors(pre,tf,fit_raw,fit_y);opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,p['neural_epochs'],eta_min=2e-5)
        cls=tf is None;positive=float(y[fitrows].mean())
        pw=torch.tensor(min(3.,max(1.,np.sqrt((1-positive)/max(positive,.01)))) if cls else 1.,device='cuda')
        best=(-float('inf'),p['neural_check_every']);stale=0;curve=[]
        for epoch in range(1,nepochs+1):
            net.train();order=rng.permutation(len(fitrows))
            for start in range(0,len(order),p['batch_size']):
                j=order[start:start+p['batch_size']];pred=net([x[j] for x in xx]);loss=train_loss(pred,yy[j],metric,pw)
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
            sched.step()
            if hold is not None and epoch%p['neural_check_every']==0:
                net.eval();query_raw=[x[hold] for x in raw]
                with torch.inference_mode():
                    q=np.concatenate([net([torch.tensor(pre[k].transform(x[start:start+256]),device='cuda') for k,x in enumerate(query_raw)]).cpu().numpy()
                        for start in range(0,len(hold),256)])
                value=utility(metric,metrics(kind(metric),y[hold],decode(q,tf)));curve.append(dict(epoch=epoch,utility=float(value)))
                if value>best[0]+1e-6:best=(value,epoch);stale=0
                else:stale+=1
                if stale>=p['neural_patience']:break
        state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};del net,xx,yy
        return state,best[1],curve
    innerpre=[Preprocess().fit(x[a]) for x in raw];innertf=transform(y[a],metric)
    for seed in p['seeds']:
        _,epoch,curve=train_one(seed,innerpre,innertf,a,p['neural_epochs'],b);epochs.append(epoch);curves.append(curve)
    pre=[Preprocess().fit(x[np.arange(len(ix))]) for x in raw];tf=transform(y,metric)
    for seed,epoch in zip(p['seeds'],epochs):
        state,_,_=train_one(seed,pre,tf,np.arange(len(ix)),epoch);states.append(state)
    return dict(family='view_attention',pre=pre,target=tf,models=states,dimensions=dims,hidden=p['hidden'],view_dropout=p['view_dropout'],
        epochs=epochs,inner_seed=splitseed,inner_fit_rows=ix[a],inner_hold_rows=ix[b],inner_curves=curves)

def boost_x(d,ix):return np.concatenate([d['molformer'][ix],d['emb'][ix],d['desc'][ix],d['maccs'][ix]],1)
def boost_fit(d,ix,y,metric,p):
    a,b,splitseed=inner_split(d,ix,y,metric);raw=boost_x(d,ix);cls=metric in ['AUROC','AUPRC'];klass=CatBoostClassifier if cls else CatBoostRegressor
    def params(depth,seed,n):return dict(iterations=n,depth=depth,learning_rate=.045,l2_leaf_reg=12,random_seed=seed,
        task_type='GPU',devices='0',gpu_ram_part=.18,thread_count=4,
        loss_function='Logloss' if cls else ('MAE' if metric=='MAE' else 'RMSE'),
        eval_metric='AUC' if metric=='AUROC' else ('PRAUC' if metric=='AUPRC' else ('MAE' if metric=='MAE' else 'RMSE')),
        bootstrap_type='Bayesian',bagging_temperature=.7,random_strength=1.,border_count=64,verbose=False,allow_writing_files=False)
    pre=Preprocess().fit(raw[a]);x=pre.transform(raw);tf=transform(y[a],metric);yy=y if cls else tf.encode(y);iterations=[]
    for depth,seed in zip(p['boost_depths'],p['seeds']):
        m=klass(**params(depth,seed,p['boost_max_iterations']));m.fit(x[a],yy[a],eval_set=(x[b],yy[b]),early_stopping_rounds=p['boost_patience'],use_best_model=True)
        iterations.append(max(80,int(m.get_best_iteration())+1))
    pre=Preprocess().fit(raw);x=pre.transform(raw);tf=transform(y,metric);yy=y if cls else tf.encode(y);models=[]
    for depth,seed,n in zip(p['boost_depths'],p['seeds'],iterations):
        m=klass(**params(depth,seed,n));m.fit(x,yy);models.append(m)
    return dict(family='mol_boost',pre=pre,target=tf,models=models,classification=cls,iterations=iterations,
        inner_seed=splitseed,inner_fit_rows=ix[a],inner_hold_rows=ix[b])

def fit(name,d,ix,y,metric):
    if not name.startswith('v16:'):return historical_fit(name,d,ix,y,metric)
    from admet_v6_runner import remote_only
    remote_only();p=recipe();family=name.split(':')[1];assert family in p['new_families']
    ix=np.asarray(ix,int);y=np.asarray(y,float);assert len(ix)==len(y)
    m=(neural_fit if family=='view_attention' else boost_fit)(d,ix,y,metric,p)
    m.update(feature_key=d['v13_feature_key'],metric=metric)
    return dict(name=name,model=m,protocol_sha256=hashlib.sha256(Path('admet_v16_protocol.json').read_bytes()).hexdigest(),
        fit_rows=ix,fit_target_sha256=hashlib.sha256(np.asarray(y,dtype='<f8').tobytes()).hexdigest())

def predict(bundle,d,ix):
    if not bundle['name'].startswith('v16:'):return historical_predict(bundle,d,ix)
    assert bundle['protocol_sha256']==hashlib.sha256(Path('admet_v16_protocol.json').read_bytes()).hexdigest()
    m=bundle['model'];assert m['feature_key']==d['v13_feature_key'];ix=np.asarray(ix,int)
    if m['family']=='mol_boost':
        x=m['pre'].transform(boost_x(d,ix));return np.mean([a.predict_proba(x)[:,1] if m['classification'] else m['target'].decode(a.predict(x)) for a in m['models']],0)
    raw=views(d,ix);x=[pp.transform(a) for pp,a in zip(m['pre'],raw)];net=ViewAttention(m['dimensions'],m['hidden'],m['view_dropout']).cuda();out=[]
    for state in m['models']:
        net.load_state_dict(state);net.eval()
        with torch.inference_mode():q=np.concatenate([net([torch.tensor(a[start:start+256],device='cuda') for a in x]).cpu().numpy() for start in range(0,len(ix),256)])
        out.append(decode(q,m['target']))
    del net
    q=np.mean(out,0);assert q.shape==(len(ix),) and np.isfinite(q).all();return q
