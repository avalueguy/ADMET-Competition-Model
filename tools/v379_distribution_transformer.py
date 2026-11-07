"""Remote-only small Transformer: native L1, ordered quantiles, ranking, EMA.

This is a new bounded comparison, not a retuning of v378 confirmation results.
Both complete schedules and both seeds run before development selection.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'4')
import hashlib
import json
import time
import joblib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import v377_regression_experts as base

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
TASKS=base.TASKS
SEEDS=(3791105,3791106)
HORIZONS=(1000,3000,6000)
BASES=(dict(name='small_ema_l1',quantiles=False),
       dict(name='small_ema_quantiles',quantiles=True))
CONFIGS=tuple(dict(name=b['name']+'_n'+str(n),base=b['name'],rounds=n) for b in BASES for n in HORIZONS)
WEIGHTS=(.1,.25,.5,.75,1.)
SETTINGS=dict(width=32,layers=1,heads=4,dropout=.3,weight_decay=.1,lr=.0004,
              minimum_lr=.00002,ema_decay=.995,ema_start=1000,
              rank_weight=.3,rank_gap=.1,quantiles=(.2,.5,.8),batch=128)
sha=base.sha
save=base.save
emit=base.emit


def signature():
    value=sha(__file__)+sha(base.__file__)+sha(base.shared.__file__)+sha(ROOT/'data_manifest.json')
    return hashlib.sha256((value+json.dumps([SETTINGS,BASES,HORIZONS,SEEDS,WEIGHTS],sort_keys=True)).encode()).hexdigest()


def setup():
    base.setup()
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')


def load():
    base.ROOT=ROOT
    base.shared.ROOT=ROOT
    return base.load()


def ctrl(t,p):return base.ctrl(t,p)
def design(d,t,p):return base.design(d,t,p)


def features(d,a):
    raw=d['fingerprints']
    fi,qi=d['mi'][a['fit']],d['mi'][a['query']]
    rms=np.sqrt((raw[fi].astype(float)**2).mean(0)).clip(.25)
    x,qx,stats=base.shared.robust(a['x'],a['qx'])
    return (np.c_[np.clip(raw[fi]/rms,0,8),x].astype(np.float32),
            np.c_[np.clip(raw[qi]/rms,0,8),qx].astype(np.float32),
            dict(rms=rms,robust=stats,extra=x.shape[1]))


class Net(nn.Module):
    def __init__(self,extra,quantiles):
        super().__init__()
        w=SETTINGS['width']
        self.quantiles=quantiles
        self.views=nn.ModuleList(nn.Linear(2048,w) for _ in range(6))
        self.numeric=nn.Linear(extra,w)
        self.position=nn.Parameter(torch.randn(7,w)*.02)
        layer=nn.TransformerEncoderLayer(w,SETTINGS['heads'],2*w,SETTINGS['dropout'],
                activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,SETTINGS['layers'],enable_nested_tensor=False)
        self.norm=nn.LayerNorm(w)
        self.head=nn.Sequential(nn.Linear(w,w),nn.GELU(),nn.Dropout(.2),nn.Linear(w,3 if quantiles else 1))

    def forward(self,x):
        tokens=[layer(x[:,j*2048:(j+1)*2048]) for j,layer in enumerate(self.views)]
        tokens.append(self.numeric(x[:,12288:]))
        h=self.norm(self.encoder(torch.stack(tokens,1)+self.position).mean(1))
        p=self.head(h)
        if not self.quantiles:return p
        center=p[:,0]
        # Ordered quantiles are modeled in the same affine native-label units.
        return torch.stack((center-F.softplus(p[:,1]),center,center+F.softplus(p[:,2])),1)


def median(p,quantiles):return p[:,1] if quantiles else p[:,0]


def objective(p,y,quantiles,ranking):
    med=median(p,quantiles)
    loss=F.l1_loss(med,y)
    if quantiles:
        low=y-p[:,0]
        high=y-p[:,2]
        lowloss=torch.maximum(.2*low,-.8*low).mean()
        highloss=torch.maximum(.8*high,-.2*high).mean()
        loss=.6*loss+.4*lowloss+.4*highloss
    if ranking:
        delta=y[:,None]-y[None]
        keep=delta.abs()>SETTINGS['rank_gap']
        if keep.any():
            pair=F.softplus(-delta.sign()*(med[:,None]-med[None]))[keep].mean()
            loss=(1-SETTINGS['rank_weight'])*loss+SETTINGS['rank_weight']*pair
    return loss


def infer(net,x):
    net.eval()
    with torch.inference_mode():
        return torch.cat([median(net(x[i:i+128]),net.quantiles) for i in range(0,len(x),128)]).float().cpu().numpy()


def update_ema(net,ema,step):
    with torch.no_grad():
        for current,target in zip(net.parameters(),ema.parameters()):
            if step<=SETTINGS['ema_start']:target.copy_(current)
            else:target.lerp_(current,1-SETTINGS['ema_decay'])


def train_curve(d,t,p,b):
    setup()
    folder=OUT/'curves'/t/p/b['name']
    path=folder/'state.joblib'
    sig=signature()
    if path.exists():
        state=joblib.load(path)
        assert state['signature']==sig
        return state
    folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    a=design(d,t,p)
    xx,qq,transform=features(d,a)
    yy,center,scale,_=base.transform(d['y'][a['fit']],False)
    x=torch.as_tensor(xx,device='cuda')
    qx=torch.as_tensor(qq,device='cuda')
    y=torch.as_tensor(yy,dtype=torch.float32,device='cuda')
    snapshots={n:[] for n in HORIZONS}
    preds={n:[] for n in HORIZONS}
    losses=[]
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng=np.random.default_rng(seed)
        net=Net(transform['extra'],b['quantiles']).cuda()
        ema=Net(transform['extra'],b['quantiles']).cuda()
        ema.load_state_dict(net.state_dict())
        ema.requires_grad_(False)
        opt=torch.optim.AdamW(net.parameters(),lr=SETTINGS['lr'],weight_decay=SETTINGS['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(HORIZONS),eta_min=SETTINGS['minimum_lr'])
        curve=[]
        for step in range(1,max(HORIZONS)+1):
            net.train()
            ids=rng.choice(len(y),min(SETTINGS['batch'],len(y)),replace=False)
            loss=objective(net(x[ids]),y[ids],b['quantiles'],t in TASKS[5:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),2.,error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            update_ema(net,ema,step)
            curve.append(float(loss.detach()))
            if step in HORIZONS:
                raw=infer(ema,qx)
                preds[step].append(base.inverse(raw.astype(float),center,scale,False,t))
                snapshots[step].append({k:v.detach().cpu().clone() for k,v in ema.state_dict().items()})
                emit('neural_horizon_done',task=t,part=p,base=b['name'],seed=seed,steps=step,
                     train_loss_last100=float(np.mean(curve[-100:])),seconds=time.monotonic()-started)
        losses.append(dict(seed=seed,first100=float(np.mean(curve[:100])),last100=float(np.mean(curve[-100:]))))
    state=dict(signature=sig,config=b,fit=a['fit'],query=a['query'],transform=transform,center=center,
               scale=scale,snapshots=snapshots,losses=losses,seconds=time.monotonic()-started)
    joblib.dump(state,path,compress=3)
    for n in HORIZONS:
        dest=OUT/'tasks'/t/p/(b['name']+'_n'+str(n))
        dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'predictions.npz',signature=sig,fit=a['fit'],query=a['query'],p=np.mean(preds[n],0))
        save(dest/'metadata.json',dict(signature=sig,base=b['name'],rounds=n,
             model_path=path.relative_to(ROOT).as_posix(),model_sha256=sha(path),prediction_sha256=sha(dest/'predictions.npz')))
    return state


def predict(d,t,p,c):
    path=OUT/'tasks'/t/p/c['name']/'predictions.npz'
    if not path.exists():train_curve(d,t,p,next(b for b in BASES if b['name']==c['base']))
    z=np.load(path)
    assert str(z['signature'])==signature()
    return z['p']


def replay(d,t,p,c):
    a=design(d,t,p)
    state=joblib.load(OUT/'curves'/t/p/c['base']/'state.joblib')
    assert state['signature']==signature()
    for key in ('fit','query'):np.testing.assert_array_equal(a[key],state[key])
    _,qq,transform=features(d,a)
    np.testing.assert_array_equal(transform['rms'],state['transform']['rms'])
    assert transform['extra']==state['transform']['extra']
    qx=torch.as_tensor(qq,device='cuda')
    predictions=[]
    for weights in state['snapshots'][c['rounds']]:
        net=Net(transform['extra'],state['config']['quantiles']).cuda()
        net.load_state_dict(weights)
        predictions.append(base.inverse(infer(net,qx).astype(float),state['center'],state['scale'],False,t))
    return np.mean(predictions,0)


def freeze():
    save(OUT/'protocol.json',dict(signature=signature(),tasks=TASKS,bases=BASES,horizons=HORIZONS,
         seeds=SEEDS,weights=WEIGHTS,settings=SETTINGS,all_fitting_remote=True,external_data=False,
         external_weights=False,test_labels_used=False,
         rationale='Complete v377/v378 curves motivate regularization and rank-aligned native objectives, not larger unchecked models.',
         architecture='6 fingerprint tokens+numeric token, 1 layer width32; ordered 0.2/0.5/0.8 quantiles versus median L1; identical EMA schedule.',
         ranking='Four Spearman endpoints use 70percent native-unit scaled regression loss and 30percent pairwise logistic rank loss. Updates still require MAE and rank gains.',
         selection='Entire6000steps and bothseeds for each curve;1000/3000/6000snapshots. Unique maxmin fold0/1; fold2/valid confirmation without fallback.',
         limitations='Historical control partly proxy; rankMAE calibrated reference; public labels reused. No score promise. Ordered intervals are auxiliary estimates, not calibrated coverage claims.'))
