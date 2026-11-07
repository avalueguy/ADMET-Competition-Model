"""Fixed-horizon feature Transformer and molecular retrieval, remote CUDA only."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'4')
import hashlib,json,time,math
import joblib,numpy as np,torch
from torch import nn
import torch.nn.functional as F
import v377_regression_experts as base

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
TASKS=base.TASKS
SEEDS=(3781105,3781106)
HORIZONS=(800,2400,4800)
BASES=(dict(name='feature_transformer',retrieval=False),dict(name='retrieval_transformer',retrieval=True))
CONFIGS=tuple(dict(name=b['name']+'_n'+str(n),base=b['name'],rounds=n) for b in BASES for n in HORIZONS)
WEIGHTS=(.1,.25,.5,.75,1.)
sha=base.sha;save=base.save;emit=base.emit


def signature():return hashlib.sha256((sha(__file__)+sha(base.__file__)+sha(base.shared.__file__)+sha(ROOT/'data_manifest.json')+json.dumps([BASES,HORIZONS,SEEDS,WEIGHTS],sort_keys=True)).encode()).hexdigest()
def setup():
 base.setup();torch.backends.cuda.matmul.allow_tf32=True;torch.set_float32_matmul_precision('high')
def load():
 base.ROOT=ROOT;base.shared.ROOT=ROOT;return base.load()
def ctrl(t,p):return base.ctrl(t,p)
def design(d,t,p):return base.design(d,t,p)


def features(d,a):
 fp=d['fingerprints'];fi=d['mi'][a['fit']];qi=d['mi'][a['query']]
 rms=np.sqrt((fp[fi].astype(float)**2).mean(0)).clip(.25)
 f=np.clip(fp[fi]/rms,0,8).astype(np.float32);q=np.clip(fp[qi]/rms,0,8).astype(np.float32)
 x,qx,stats=base.shared.robust(a['x'],a['qx'])
 x=x.astype(np.float32);qx=qx.astype(np.float32)
 return np.c_[f,x],np.c_[q,qx],dict(rms=rms,robust=stats,extra=x.shape[1])


def neighbors(d,a,count=32):
 fit,query=a['fit'],a['query'];raw=d['fingerprints'][:,:4096]
 ref=torch.as_tensor(raw[d['mi'][fit]],device='cuda');norm=ref.square().sum(1)
 result=[]
 for rows,training in ((fit,True),(query,False)):
  ids=[];sims=[]
  for lo in range(0,len(rows),128):
   rr=rows[lo:lo+128];q=torch.as_tensor(raw[d['mi'][rr]],device='cuda');dot=q@ref.T
   sim=(dot/(q.square().sum(1)[:,None]+norm[None]-dot).clamp_min(1e-8)).clamp(0,1).cpu().numpy()
   if training:
    excluded=(d['rg'][rr,None]==d['rg'][fit][None])|(d['canonical'][d['mi'][rr],None]==d['canonical'][d['mi'][fit]][None])
    sim[excluded]=-1
   ix=np.argsort(-sim,axis=1,kind='stable')[:,:min(count,len(fit))];values=np.take_along_axis(sim,ix,1)
   assert (values>=0).all(),'Not enough group-disjoint contexts'
   ids.append(ix);sims.append(values)
  result.append((np.concatenate(ids),np.concatenate(sims)))
 return result


class Net(nn.Module):
 def __init__(self,extra,retrieval):
  super().__init__();self.retrieval=retrieval;self.extra=extra
  self.views=nn.ModuleList(nn.Linear(2048,64) for _ in range(6));self.numeric=nn.Linear(extra,64)
  self.position=nn.Parameter(torch.randn(7,64)*.02)
  layer=nn.TransformerEncoderLayer(64,4,128,.15,activation='gelu',batch_first=True,norm_first=True)
  self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.norm=nn.LayerNorm(64)
  self.direct=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
  if retrieval:
   self.key=nn.Linear(64,32);self.value=nn.Sequential(nn.Linear(65,64),nn.GELU(),nn.Linear(64,64))
   self.head=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1));self.label_weight=nn.Parameter(torch.tensor(.3))
 def encode(self,x):
  tokens=[layer(x[:,j*2048:(j+1)*2048]) for j,layer in enumerate(self.views)]+[self.numeric(x[:,12288:])]
  return self.norm(self.encoder(torch.stack(tokens,1)+self.position).mean(1))
 def combine(self,h,ref,labels,similarity):
  if not self.retrieval:return self.direct(h).squeeze(1)
  delta=self.key(h)[:,None]-self.key(ref)
  logits=-delta.square().mean(-1)+2*similarity.clamp_min(1e-4).log()
  if self.training:
   keep=torch.rand_like(labels)>.15;keep[:,0]=True;logits=logits.masked_fill(~keep,-1e9)
  weights=logits.softmax(1);v=self.value(torch.cat([ref-h[:,None],labels[:,:,None]],-1));ctx=(v*weights[:,:,None]).sum(1)
  return self.direct(h).squeeze(1)+self.head(torch.cat([h,ctx],1)).squeeze(1)+self.label_weight*(weights*labels).sum(1)


def infer(net,x,qx,labels,query_neighbors,batch=128):
 net.eval();outputs=[]
 with torch.inference_mode():
  reference=torch.cat([net.encode(x[lo:lo+batch]) for lo in range(0,len(x),batch)]) if net.retrieval else None
  for lo in range(0,len(qx),batch):
   h=net.encode(qx[lo:lo+batch])
   if net.retrieval:
    ix,sim=query_neighbors;ids=ix[lo:lo+batch];p=net.combine(h,reference[ids],labels[ids],sim[lo:lo+batch])
   else:p=net.direct(h).squeeze(1)
   outputs.append(p.float().cpu().numpy())
 return np.concatenate(outputs)


def train_curve(d,t,p,b):
 setup();folder=OUT/'curves'/t/p/b['name'];folder.mkdir(parents=True,exist_ok=True);path=folder/'state.joblib';sig=signature()
 if path.exists():s=joblib.load(path);assert s['signature']==sig;return s
 started=time.monotonic();a=design(d,t,p);xx,qq,transform=features(d,a);yy,center,scale,_=base.transform(d['y'][a['fit']],False)
 x=torch.as_tensor(xx,device='cuda');qx=torch.as_tensor(qq,device='cuda');y=torch.as_tensor(yy,dtype=torch.float32,device='cuda')
 if b['retrieval']:
  ns=neighbors(d,a);train_ix=torch.as_tensor(ns[0][0],device='cuda');train_sim=torch.as_tensor(ns[0][1],device='cuda');qn=tuple(torch.as_tensor(v,device='cuda') for v in ns[1])
 else:ns=None;qn=None
 snapshots={n:[] for n in HORIZONS};preds={n:[] for n in HORIZONS};losses=[]
 for seed in SEEDS:
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);net=Net(transform['extra'],b['retrieval']).cuda()
  opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.04);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(HORIZONS),eta_min=.00002)
  # Context embeddings refreshed each40steps; the previous fixed bank is detached.
  bank=None;curve=[]
  for step in range(1,max(HORIZONS)+1):
   if b['retrieval'] and (bank is None or (step-1)%40==0):
    net.eval()
    with torch.no_grad():bank=torch.cat([net.encode(x[i:i+128]) for i in range(0,len(x),128)]).detach()
   net.train();ids=rng.choice(len(y),min(128,len(y)),replace=False);h=net.encode(x[ids])
   if b['retrieval']:
    ix=train_ix[ids];pred=net.combine(h,bank[ix],y[ix],train_sim[ids])
   else:pred=net.direct(h).squeeze(1)
   # Smooth-to-L1 curriculum warmup, then optimize exact native-scaled L1.
   loss=F.smooth_l1_loss(pred,y[ids],beta=.25) if step<=400 else F.l1_loss(pred,y[ids])
   opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.,error_if_nonfinite=True);opt.step();scheduler.step();curve.append(float(loss.detach()))
   if step in HORIZONS:
    raw=infer(net,x,qx,y,qn);preds[step].append(base.inverse(raw.astype(float),center,scale,False,t))
    snapshots[step].append({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
    emit('neural_horizon_done',task=t,part=p,base=b['name'],seed=seed,steps=step,train_loss_last100=float(np.mean(curve[-100:])),seconds=time.monotonic()-started)
  losses.append(dict(seed=seed,first100=float(np.mean(curve[:100])),last100=float(np.mean(curve[-100:]))))
 s=dict(signature=sig,config=b,fit=a['fit'],query=a['query'],transform=transform,center=center,scale=scale,snapshots=snapshots,neighbors=ns,losses=losses,seconds=time.monotonic()-started)
 joblib.dump(s,path,compress=3)
 for n in HORIZONS:
  dest=OUT/'tasks'/t/p/(b['name']+'_n'+str(n));dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'predictions.npz',signature=sig,fit=a['fit'],query=a['query'],p=np.mean(preds[n],0))
  save(dest/'metadata.json',dict(signature=sig,base=b['name'],rounds=n,model_path=path.relative_to(ROOT).as_posix(),model_sha256=sha(path),prediction_sha256=sha(dest/'predictions.npz')))
 return s


def predict(d,t,p,c):
 path=OUT/'tasks'/t/p/c['name']/'predictions.npz'
 if not path.exists():train_curve(d,t,p,next(b for b in BASES if b['name']==c['base']))
 z=np.load(path);assert str(z['signature'])==signature();return z['p']


def replay(d,t,p,c):
 a=design(d,t,p);s=joblib.load(OUT/'curves'/t/p/c['base']/'state.joblib');assert s['signature']==signature();xx,qq,trans=features(d,a)
 np.testing.assert_array_equal(trans['rms'],s['transform']['rms']);assert trans['extra']==s['transform']['extra']
 x=torch.as_tensor(xx,device='cuda');qx=torch.as_tensor(qq,device='cuda');y=torch.as_tensor((d['y'][a['fit']]-s['center'])/s['scale'],dtype=torch.float32,device='cuda')
 qn=tuple(torch.as_tensor(v,device='cuda') for v in s['neighbors'][1]) if s['config']['retrieval'] else None
 result=[]
 for weights in s['snapshots'][c['rounds']]:
  net=Net(trans['extra'],s['config']['retrieval']).cuda();net.load_state_dict(weights);raw=infer(net,x,qx,y,qn);result.append(base.inverse(raw.astype(float),s['center'],s['scale'],False,t))
 return np.mean(result,0)


def freeze():
 save(OUT/'protocol.json',dict(signature=signature(),tasks=TASKS,bases=BASES,horizons=HORIZONS,seeds=SEEDS,weights=WEIGHTS,
  goal='Nine native-unit relativeMAE gains average5percent, separately scored5; ranking requires rankgain.',
  architecture='6fingerprinttokens+numeric,2Transformerlayers64width4heads; optional32labelreferences with chemistry similarity and learned key/value.',
  curriculum='400smoothL1warmup then exactL1;4800steps eachseed completed even if800/2400poor.',
  retrieval='Fitting query self/canonical/scaffoldexcluded. Neighbors structureonly. Detachedreferencefeaturebank refreshed40steps; fitlabels never used as features of same-group query.',
  selection='Same fold0/1unique maxmin, then fold2/valid without fallback.',all_fitting_remote=True,external_weights=False,
  limitations='Historical proxy and calibrated rankMAE reference; repeatedlyusedpublicvalidation; bank approximation is not exact end-to-end TabR.'))
