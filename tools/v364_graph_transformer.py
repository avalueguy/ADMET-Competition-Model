"""Stage1: fold-safe atom/bond Transformer, remote CUDA training only.

The learner sees only official fitting rows. Existing controls remain proxies
where original complete historical OOF is unavailable. No external weights.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, platform, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger, rdBase
from scipy.stats import rankdata, spearmanr
from scipy.special import ndtri, expit
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
TASKS=('caco2_wang','clearance_hepatocyte_az','cyp2d6_veith','herg','lipophilicity_astrazeneca','solubility_aqsoldb')
SEEDS=(3641105,3641106)
CONFIG=dict(layers=6,width=192,heads=6,dropout=.15,adapter_rank=8,pretrain_steps=1600,
            pretrain_batch=64,finetune_min_steps=700,finetune_max_steps=1600,
            finetune_epochs=65,finetune_batch=64,lr=.0004,weight_decay=.02)
MODES=('scratch','selfsupervised','family_transfer')
FAMILY={
 'caco2_wang':('lipophilicity_astrazeneca','solubility_aqsoldb','pgp_broccatelli','hia_hou'),
 'clearance_hepatocyte_az':('clearance_microsome_az','half_life_obach','vdss_lombardo'),
 'cyp2d6_veith':('cyp2c9_veith','cyp3a4_veith','cyp2d6_substrate_carbonmangels'),
 'herg':('ames','dili','pgp_broccatelli'),
 'lipophilicity_astrazeneca':('solubility_aqsoldb','caco2_wang','ppbr_az'),
 'solubility_aqsoldb':('lipophilicity_astrazeneca','caco2_wang','ppbr_az'),
}
RDLogger.DisableLog('rdApp.*')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def emit(stage,**x):print(json.dumps(dict(stage=stage,**x),ensure_ascii=False),flush=True)
def signature():return hashlib.sha256((sha(__file__)+sha(ROOT/'data_manifest.json')+json.dumps(CONFIG,sort_keys=True)).encode()).hexdigest()
def setup():
 assert platform.system()=='Linux' and torch.cuda.is_available(),'All training must run on authorized Linux CUDA host'
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 torch.set_float32_matmul_precision('high')
def load():
 z=np.load(ROOT/'official_rows.npz',allow_pickle=False);d={k:z[k] for k in z.files};d.update(json.loads((ROOT/'data_manifest.json').read_text(encoding='utf-8')))
 d['rg']=d['groups'][d['mi']];return d
def parse(sm):
 mol=Chem.MolFromSmiles(sm)
 if mol is None:
  mol=Chem.MolFromSmiles(sm,sanitize=False);assert mol is not None
  mol.UpdatePropertyCache(strict=False)
  flag=Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,catchErrors=True)
  assert flag==Chem.SanitizeFlags.SANITIZE_NONE
 return mol
def graph_one(sm):
 m=parse(sm);n=m.GetNumAtoms();a=np.zeros((n,9),np.int64)
 for i,atom in enumerate(m.GetAtoms()):
  a[i]=[min(atom.GetAtomicNum(),120),min(atom.GetDegree(),10),min(max(atom.GetFormalCharge(),-5),5)+6,
        int(atom.GetIsAromatic())+1,min(int(atom.GetHybridization()),8)+1,min(int(atom.GetChiralTag()),4)+1,
        min(atom.GetTotalNumHs(),8)+1,int(atom.IsInRing())+1,min(atom.GetIsotope(),299)+1]
 b=np.zeros((n,n),np.uint8)
 for bond in m.GetBonds():
  val={Chem.BondType.SINGLE:1,Chem.BondType.DOUBLE:2,Chem.BondType.TRIPLE:3,Chem.BondType.AROMATIC:4}.get(bond.GetBondType(),5)
  i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx();b[i,j]=b[j,i]=val
 dist=Chem.GetDistanceMatrix(m);distance=np.where(dist>1e6,17,np.minimum(dist,16)).astype(np.uint8)
 return dict(atom=a,bond=b,distance=distance,n=n)
def prepare_graphs(d):
 path=ROOT/'graphs.pt';key=hashlib.sha256((sha(__file__)+str(rdBase.rdkitVersion)+str(len(d['smiles']))).encode()).hexdigest()
 if path.exists():
  x=torch.load(path,weights_only=False,map_location='cpu');assert x['key']==key;return x['graphs']
 graphs=[]
 for i,sm in enumerate(d['smiles']):
  graphs.append(graph_one(str(sm)))
  if i%5000==0:emit('graphs',done=i,total=len(d['smiles']))
 torch.save(dict(key=key,graphs=graphs),path)
 save(ROOT/'graph_audit.json',dict(status='PASS',molecules=len(graphs),rdkit=rdBase.rdkitVersion,maximum_atoms=max(g['n'] for g in graphs),no_truncation=True,source_sha256=sha(__file__)))
 return graphs
def batch(graphs,ids,device='cuda',mask_probability=0.,permutation=False,rng=None):
 rng=np.random.default_rng() if rng is None else rng
 gs=[graphs[int(i)] for i in ids];n=max(g['n'] for g in gs);bs=len(gs)
 aa=np.zeros((bs,n+1,9),np.int64);bb=np.zeros((bs,n+1,n+1),np.int64);dd=np.full_like(bb,17);valid=np.zeros((bs,n+1),bool)
 aa[:,0,0]=122;dd[:,0,:]=18;dd[:,:,0]=18;dd[:,0,0]=0;valid[:,0]=True
 for i,g in enumerate(gs):
  order=rng.permutation(g['n']) if permutation else np.arange(g['n']);stop=g['n']+1
  aa[i,1:stop]=g['atom'][order];bb[i,1:stop,1:stop]=g['bond'][np.ix_(order,order)];dd[i,1:stop,1:stop]=g['distance'][np.ix_(order,order)];valid[i,1:stop]=True
 truth=aa.copy();atom_mask=(rng.random((bs,n+1))<mask_probability)&valid;atom_mask[:,0]=False
 if mask_probability:
  for i,g in enumerate(gs):
   if not atom_mask[i].any():atom_mask[i,rng.integers(1,g['n']+1)]=True
  aa[atom_mask]=0;aa[:,:,0][atom_mask]=121
 edge_mask=(rng.random(bb.shape)<mask_probability)&(bb>0)
 edge_mask=edge_mask|edge_mask.transpose(0,2,1);edge_truth=bb.copy();bb[edge_mask]=6
 return {k:torch.as_tensor(v,device=device) for k,v in dict(atom=aa,bond=bb,distance=dd,valid=valid,truth=truth,atom_mask=atom_mask,edge_mask=edge_mask,edge_truth=edge_truth).items()}

class Adapter(nn.Module):
 def __init__(self,width,rank):
  super().__init__();self.down=nn.Linear(width,rank,bias=False);self.up=nn.Linear(rank,width,bias=False);nn.init.zeros_(self.up.weight)
 def forward(self,x):return self.up(F.gelu(self.down(x)))
class Block(nn.Module):
 def __init__(self):
  super().__init__();w=CONFIG['width'];self.h=CONFIG['heads'];self.norm1=nn.LayerNorm(w);self.qkv=nn.Linear(w,w*3);self.proj=nn.Linear(w,w)
  self.norm2=nn.LayerNorm(w);self.ff=nn.Sequential(nn.Linear(w,4*w),nn.GELU(),nn.Dropout(CONFIG['dropout']),nn.Linear(4*w,w))
  self.drop=nn.Dropout(CONFIG['dropout']);self.adapter=Adapter(w,CONFIG['adapter_rank'])
 def forward(self,x,bias,valid):
  b,n,w=x.shape;q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,self.h,w//self.h).permute(2,0,3,1,4).unbind(0)
  mask=bias.masked_fill(~valid[:,None,None,:],float('-inf'))
  out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=CONFIG['dropout'] if self.training else 0.)
  x=x+self.drop(self.proj(out.transpose(1,2).reshape(b,n,w)));z=self.norm2(x);return x+self.drop(self.ff(z))+self.adapter(z)
class Encoder(nn.Module):
 def __init__(self):
  super().__init__();w=CONFIG['width'];h=CONFIG['heads'];sizes=(123,11,12,3,10,6,10,3,301)
  self.atom=nn.ModuleList(nn.Embedding(s,w,padding_idx=0) for s in sizes);self.distance=nn.Embedding(19,h);self.bond=nn.Embedding(7,h,padding_idx=0)
  self.blocks=nn.ModuleList(Block() for _ in range(CONFIG['layers']));self.norm=nn.LayerNorm(w)
 def forward(self,b):
  x=sum(emb(b['atom'][:,:,i]) for i,emb in enumerate(self.atom));bias=(self.distance(b['distance'])+self.bond(b['bond'])).permute(0,3,1,2)
  for layer in self.blocks:x=layer(x,bias,b['valid'])
  x=self.norm(x);mask=b['valid'].clone();mask[:,0]=False
  mean=(x*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
  return x,torch.cat([x[:,0],mean],1)
class Pretrain(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=Encoder();w=CONFIG['width'];self.element=nn.Linear(w,121);self.charge=nn.Linear(w,12);self.aromatic=nn.Linear(w,3)
  self.edge=nn.Sequential(nn.Linear(w,64),nn.GELU(),nn.Linear(64,6));self.project=nn.Sequential(nn.Linear(w*2,w),nn.GELU(),nn.Linear(w,w))
 def forward(self,b):
  x,pool=self.encoder(b);mask=b['atom_mask'];target=b['truth']
  loss=F.cross_entropy(self.element(x[mask]),target[:,:,0][mask])+.25*F.cross_entropy(self.charge(x[mask]),target[:,:,2][mask])+.25*F.cross_entropy(self.aromatic(x[mask]),target[:,:,3][mask])
  indices=b['edge_mask'].nonzero()
  if len(indices):
   i,j,k=indices.unbind(1);loss=loss+.4*F.cross_entropy(self.edge(x[i,j]+x[i,k]),b['edge_truth'][i,j,k])
  return loss,F.normalize(self.project(pool),dim=1)
class Predictor(nn.Module):
 def __init__(self,desc_dim):
  super().__init__();self.encoder=Encoder();w=CONFIG['width'];self.desc=nn.Sequential(nn.Linear(desc_dim,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.2))
  self.head=nn.Sequential(nn.Linear(w*2+64,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
 def forward(self,b,desc):return self.head(torch.cat([self.encoder(b)[1],self.desc(desc)],1)).squeeze(1)
def ctrl(task,part):
 z=np.load(ROOT/'controls'/task/(part+'.npz'));return {k:z[k] for k in z.files}
def global_fit(d,part):
 limit=2 if part=='final' else 1;rows=np.flatnonzero(d['split']<limit)
 if part!='final':
  query=np.unique(np.concatenate([ctrl(t,part)['query'] for t in TASKS]));blocked=np.unique(d['rg'][query]);rows=rows[~np.isin(d['rg'][rows],blocked)]
  assert not set(d['rg'][rows])&set(blocked)
 return rows
def pretrain(d,graphs,part,seed):
 setup();folder=OUT/'encoders'/part/str(seed);modelpath=folder/'pretrained.pt';sig=signature()
 if modelpath.exists():
  state=torch.load(modelpath,map_location='cpu',weights_only=False);assert state['signature']==sig;return state['encoder']
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
 rows=global_fit(d,part);ids=np.unique(d['mi'][rows]);folder.mkdir(parents=True,exist_ok=True)
 net=Pretrain().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.02);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,CONFIG['pretrain_steps'],eta_min=.00003)
 start=time.monotonic();losses=[]
 emit('pretrain_start',part=part,seed=seed,molecules=len(ids),parameters=sum(p.numel() for p in net.parameters()))
 for step in range(CONFIG['pretrain_steps']):
  picked=rng.choice(ids,CONFIG['pretrain_batch'],replace=False);b=batch(graphs,picked,mask_probability=.15,permutation=True,rng=rng)
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   loss,z=net(b)
   if step%2==0:
    other=batch(graphs,picked,mask_probability=.15,permutation=True,rng=rng);loss2,z2=net(other)
    logits=z@z2.T/.15;target=torch.arange(len(z),device='cuda');cons=.5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target));loss=.5*(loss+loss2)+.15*cons
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();sch.step();losses.append(float(loss.detach()))
  if (step+1)%100==0:emit('pretrain_progress',part=part,seed=seed,step=step+1,loss=float(np.mean(losses[-100:])),seconds=time.monotonic()-start)
 state={k:v.detach().cpu() for k,v in net.encoder.state_dict().items()};torch.save(dict(signature=sig,encoder=state,fit_rows=rows,fit_molecules=ids,config=CONFIG),modelpath)
 save(folder/'metadata.json',dict(status='COMPLETE',signature=sig,part=part,seed=seed,fit_rows=len(rows),fit_molecules=len(ids),seconds=time.monotonic()-start,first100=float(np.mean(losses[:100])),last100=float(np.mean(losses[-100:])),checkpoint_sha256=sha(modelpath)))
 del net,opt;torch.cuda.empty_cache();return state
def normalize_desc(d,fit):
 raw=d['descriptor'];rr=raw[np.unique(d['mi'][fit])].astype(float);med=np.nan_to_num(np.nanmedian(rr,axis=0));rr=np.where(np.isfinite(rr),rr,med);mean=rr.mean(0);std=rr.std(0).clip(.05)
 x=np.c_[np.clip((np.where(np.isfinite(raw),raw,med)-mean)/std,-6,6),~np.isfinite(raw)].astype(np.float32)
 return x,dict(median=med,mean=mean,std=std)
def target(y,metric):
 center=float(np.median(y)) if metric=='MAE' else 0.;scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if metric=='MAE' else 1.
 values=(y-center)/scale if metric=='MAE' else ndtri(np.clip((rankdata(y)-.5)/len(y),.002,.998)) if metric=='Spearman' else y
 return values.astype(np.float32),dict(center=center,scale=scale)
def objective(pred,y,metric):
 if metric in ('AUROC','AUPRC'):
  loss=F.binary_cross_entropy_with_logits(pred,y);a,z=pred[y>.5],pred[y<.5]
  if len(a) and len(z):loss=loss+.2*F.softplus(z[:,None]-a[None]).mean()
  return loss
 if metric=='MAE':return (pred-y).abs().mean()
 delta=y[:,None]-y[None];mask=delta.abs()>.15;loss=.7*(pred-y).square().mean()
 if mask.any():loss=loss+.3*F.softplus(-delta.sign()*(pred[:,None]-pred[None]))[mask].mean()
 return loss
def configure(net,mode,task):
 if mode=='scratch':return
 for p in net.encoder.parameters():p.requires_grad=False
 if task in ('cyp2d6_veith','lipophilicity_astrazeneca','solubility_aqsoldb'):
  for layer in net.encoder.blocks[-2:]:
   for p in layer.parameters():p.requires_grad=True
 for layer in net.encoder.blocks:
  for p in layer.adapter.parameters():p.requires_grad=True
 for p in net.encoder.norm.parameters():p.requires_grad=True
def fit_updates(net,opt,sch,d,graphs,fit,tx,desc,metric,steps,rng,label):
 losses=[];start=time.monotonic()
 for step in range(steps):
  net.train();ix=rng.choice(len(fit),CONFIG['finetune_batch'],replace=len(fit)<CONFIG['finetune_batch']);rows=fit[ix];b=batch(graphs,d['mi'][rows],rng=rng)
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(b,desc[d['mi'][rows]]);loss=objective(pred,tx[ix],metric)
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();sch.step();losses.append(float(loss.detach()))
  if (step+1)%200==0:emit('fit_progress',label=label,step=step+1,steps=steps,loss=float(np.mean(losses[-100:])),seconds=time.monotonic()-start)
 return dict(seconds=time.monotonic()-start,first100=float(np.mean(losses[:100])),last100=float(np.mean(losses[-100:])))
def infer(net,d,graphs,q,desc,device='cuda',chunk=64):
 net.eval();out=[];tf32=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
 with torch.inference_mode():
  for i in range(0,len(q),chunk):
   rows=q[i:i+chunk];b=batch(graphs,d['mi'][rows],device=device);out.append(net(b,desc[d['mi'][rows]]).detach().float().cpu().numpy())
 torch.backends.cuda.matmul.allow_tf32=tf32
 return np.concatenate(out).astype(float)
def transfer(d,graphs,net,fit,task,part,seed,desc,rng):
 # Family initialization uses separate heads; outer query groups excluded globally.
 allowed=global_fit(d,part);family=[t for t in FAMILY[task] if t in d['tasks']]
 original=copy.deepcopy(net.head.state_dict());stats=[]
 # Equal short auxiliary budgets, frozen before any evaluation.
 for t in family:
  ix=allowed[d['tid'][allowed]==d['tasks'].index(t)]
  if len(ix)<10:continue
  torch.manual_seed(seed+10+d['tasks'].index(t));net.head.apply(lambda m: m.reset_parameters() if isinstance(m,nn.Linear) else None)
  yy,_=target(d['y'][ix],d['metrics'][d['tasks'].index(t)]);tx=torch.as_tensor(yy,device='cuda')
  opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.00015,weight_decay=.02);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,120)
  stats.append(dict(task=t,**fit_updates(net,opt,sch,d,graphs,ix,tx,desc,d['metrics'][d['tasks'].index(t)],120,rng,f'family:{task}:{t}:{part}:{seed}')))
 net.head.load_state_dict(original);return stats
def predict(d,graphs,task,part,mode):
 setup();folder=OUT/'tasks'/task/part/mode;file=folder/'predictions.npz';sig=signature()
 if file.exists():
  z=np.load(file);assert str(z['signature'])==sig;return z['p']
 a=ctrl(task,part);fit,q=a['fit'],a['query'];k=d['tasks'].index(task);metric=d['metrics'][k]
 assert np.all(d['split'][fit]<2) and not set(fit)&set(q)
 if part!='final':assert not set(d['rg'][fit])&set(d['rg'][q])
 xx,ds=normalize_desc(d,fit);desc=torch.as_tensor(xx,device='cuda');yy,ys=target(d['y'][fit],metric);tx=torch.as_tensor(yy,device='cuda')
 folder.mkdir(parents=True,exist_ok=True);steps=int(np.clip(math.ceil(len(fit)/CONFIG['finetune_batch'])*CONFIG['finetune_epochs'],CONFIG['finetune_min_steps'],CONFIG['finetune_max_steps']));members=[];metadata=[]
 for seed in SEEDS:
  path=folder/f'model_{seed}.pt'
  if path.exists():
   state=torch.load(path,map_location='cpu',weights_only=False);assert state['signature']==sig;np.testing.assert_array_equal(state['fit'],fit);np.testing.assert_array_equal(state['query'],q)
   net=Predictor(xx.shape[1]).cuda();net.load_state_dict(state['weights']);members.append(infer(net,d,graphs,q,desc));metadata.append(dict(seed=seed,checkpoint_sha256=sha(path),resumed=True));del net;continue
  initial=pretrain(d,graphs,part,seed) if mode!='scratch' else None
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);net=Predictor(xx.shape[1]).cuda()
  if initial is not None:net.encoder.load_state_dict(initial)
  configure(net,mode,task);extra=transfer(d,graphs,net,fit,task,part,seed,desc,rng) if mode=='family_transfer' else []
  opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay']);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00002)
  emit('fit_start',task=task,part=part,mode=mode,seed=seed,steps=steps,trainable=sum(p.numel() for p in net.parameters() if p.requires_grad))
  stats=fit_updates(net,opt,sch,d,graphs,fit,tx,desc,metric,steps,rng,f'{task}:{part}:{mode}:{seed}');raw=infer(net,d,graphs,q,desc)
  state=dict(signature=sig,config=CONFIG,weights={k:v.detach().cpu() for k,v in net.state_dict().items()},fit=fit,query=q,descriptor=ds,target=ys,task=task,part=part,mode=mode,seed=seed)
  torch.save(state,path);reloaded=Predictor(xx.shape[1]).cuda();reloaded.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['weights']);again=infer(reloaded,d,graphs,q,desc)
  np.testing.assert_allclose(raw,again,atol=1e-6,rtol=1e-6);members.append(raw);metadata.append(dict(seed=seed,**stats,checkpoint_sha256=sha(path),family=extra,reload_max_error=float(np.max(np.abs(raw-again)))))
  del net,reloaded,opt;torch.cuda.empty_cache()
 raw=np.mean(members,axis=0);p=expit(raw) if metric in ('AUROC','AUPRC') else raw*ys['scale']+ys['center']
 assert np.isfinite(p).all();np.savez_compressed(file,signature=sig,fit=fit,query=q,p=p,raw_members=members)
 save(folder/'metadata.json',dict(status='COMPLETE',signature=sig,task=task,part=part,mode=mode,metric=metric,steps=steps,seeds=metadata,fit_rows=len(fit),query_rows=len(q),target=ys,source_sha256=sha(__file__),predictions_sha256=sha(file)))
 emit('fit_done',task=task,part=part,mode=mode,rows=len(q));return p
def score(m,y,p):return float(mean_absolute_error(y,p) if m=='MAE' else spearmanr(y,p).statistic if m=='Spearman' else roc_auc_score(y,p) if m=='AUROC' else average_precision_score(y,p))
def gain(m,y,a,p):
 old,new=score(m,y,a),score(m,y,p);return (old-new)/old if m=='MAE' else new-old
def mix(m,a,p,w):return (1-w)*a+w*p if m=='MAE' else (1-w)*rankdata(a)/len(a)+w*rankdata(p)/len(p)
def threshold(m):return .012 if m=='MAE' else .008 if m=='Spearman' else .003
def smoke(d,graphs):
 setup();torch.manual_seed(364);rng=np.random.default_rng(364);ids=np.array([1,7,39,200]);net=Pretrain().cuda();before=net.element.weight.detach().clone();opt=torch.optim.AdamW(net.parameters(),lr=.001)
 b=batch(graphs,ids,mask_probability=.2,rng=rng);loss,z=net(b);loss.backward();assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None);opt.step();assert not torch.equal(before,net.element.weight)
 net.eval();torch.backends.cuda.matmul.allow_tf32=False
 with torch.inference_mode():
  aa=net.encoder(batch(graphs,ids))[1];bb=net.encoder(batch(graphs,ids,permutation=True,rng=rng))[1];error=float((aa-bb).abs().max());assert error<.0001,error
  single=net.encoder(batch(graphs,ids[:1]))[1];batch_error=float((single-aa[:1]).abs().max());assert batch_error<.0001,batch_error
 # Check direction and magnitude of the four task losses with known examples.
 loss_checks={}
 for metric in ('MAE','Spearman','AUROC','AUPRC'):
  y=torch.tensor([-1.,0.,1.,2.] if metric in ('MAE','Spearman') else [0.,0.,1.,1.],device='cuda')
  good=y if metric in ('MAE','Spearman') else torch.tensor([-3.,-2.,2.,3.],device='cuda')
  bad=good.flip(0);a=float(objective(good,y,metric));b=float(objective(bad,y,metric));assert a<b,(metric,a,b);loss_checks[metric]=dict(good=a,bad=b)
 # Strict train/query isolation is checked for all four development parts.
 overlap=[]
 for part in ('fold0','fold1','fold2','valid'):
  fit=global_fit(d,part);query=np.unique(np.concatenate([ctrl(t,part)['query'] for t in TASKS]));assert not set(d['rg'][fit])&set(d['rg'][query]);overlap.append(dict(part=part,fit=len(fit),query=len(query),group_overlap=0))
 # Throughput calibration uses structures and synthetic labels, never query outcomes.
 del net,opt;torch.cuda.empty_cache();bench=Predictor(d['descriptor'].shape[1]*2).cuda();opt=torch.optim.AdamW(bench.parameters(),lr=.0004)
 fit=global_fit(d,'fold0');pool=np.unique(d['mi'][fit]);times=[];torch.cuda.reset_peak_memory_stats()
 for step in range(25):
  ix=rng.choice(pool,64,replace=False);t0=time.monotonic();b=batch(graphs,ix);opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=bench(b,torch.zeros((64,d['descriptor'].shape[1]*2),device='cuda'));ll=pred.square().mean()
  ll.backward();opt.step();torch.cuda.synchronize();times.append(time.monotonic()-t0)
 timing=dict(median_seconds=float(np.median(times[5:])),p95_seconds=float(np.quantile(times[5:],.95)),peak_gpu_bytes=torch.cuda.max_memory_allocated(),steps=25,batch=64)
 save(OUT/'smoke.json',dict(status='PASS',signature=signature(),cuda_device=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in bench.parameters()),loss=float(loss.detach()),permutation_max_error=error,batch_max_error=batch_error,loss_checks=loss_checks,actual_gradient_update=True,no_truncation=True,isolation=overlap,benchmark=timing));emit('smoke_pass',permutation_max_error=error,benchmark=timing)
def freeze_protocol():
 p=OUT/'protocol.json';x=dict(signature=signature(),tasks=TASKS,config=CONFIG,seeds=SEEDS,modes=MODES,weights=[0.,.2,.45],baseline='v361:51.240',
 selection='Unique max-min fold0/1 config and weight, frozen before fold2/valid; strict threshold retained; optional experimental requires both developmental gains positive and no fallback.',
 confirmation='all4 positive, mean practical threshold, both500 paired scaffold bootstrap90 lower bounds positive for strict; all4positive with cross-zero CI is experimental only',
 source_sha256=sha(__file__),external_data=False,external_weights=False,test_labels_used=False,training_host='connect.nmb2.seetacloud.com:24365',
 target_only_control='scratch',self_supervised='masked atom/bond plus stochastic graph-view consistency',family_transfer='fixed120steps per related task, no target query labels',
 small_tasks='frozen backbone plus rank8adapters',large_tasks='last2layers plus adapters',upstream_controls='adopted v361 incremental layers exact; early OOF partly proxy')
 if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(x))
 else:save(p,x)
def train_dev(d,graphs,modes):
 freeze_protocol()
 for part in ('fold0','fold1'):
  for mode in modes:
   for task in TASKS:predict(d,graphs,task,part,mode)
def selection(d):
 result=[];records=[]
 for task in TASKS:
  k=d['tasks'].index(task);m=d['metrics'][k];choices=[]
  for mode in MODES:
   files=[OUT/'tasks'/task/p/mode/'predictions.npz' for p in ('fold0','fold1')]
   assert all(p.exists() for p in files),('Incomplete development experiment',task,mode)
   for file in files:assert str(np.load(file)['signature'])==signature()
   for w in (.2,.45):
    parts=[]
    for part,file in zip(('fold0','fold1'),files):
     a=ctrl(task,part);p=np.load(file)['p'];new=mix(m,a['p'],p,w);y=d['y'][a['query']]
     parts.append(dict(part=part,baseline=score(m,y,a['p']),expert=score(m,y,p),candidate=score(m,y,new),gain=gain(m,y,a['p'],new)))
    choices.append(dict(task=task,metric=m,mode=mode,weight=w,parts=parts,objective=min(p['gain'] for p in parts)))
  assert choices;best=max(choices,key=lambda r:(r['objective'],-r['weight'],-MODES.index(r['mode'])));records.append(dict(task=task,choices=choices,best=best))
  if best['objective']>0:result.append(dict(**best,strict_development=best['objective']>threshold(m)))
  emit('development_selection',**best)
 save(OUT/'development.json',dict(status='COMPLETE',signature=signature(),records=records))
 path=OUT/'frozen_selection.json';value=dict(signature=signature(),selected=result,all_development_complete=True,no_confirmation_fallback=True)
 if path.exists():assert json.loads(path.read_text())==value
 else:save(path,value)
def confirm(d,graphs):
 frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==signature();records=[]
 for chosen in frozen['selected']:
  t=chosen['task'];k=d['tasks'].index(t);m=d['metrics'][k];parts=[];boot=[]
  for part in ('fold0','fold1','fold2','valid'):
   p=predict(d,graphs,t,part,chosen['mode']);a=ctrl(t,part);new=mix(m,a['p'],p,chosen['weight']);y=d['y'][a['query']]
   parts.append(dict(part=part,baseline=score(m,y,a['p']),candidate=score(m,y,new),gain=gain(m,y,a['p'],new)))
   if part in ('fold2','valid'):
    groups=d['rg'][a['query']];ug=np.unique(groups);ixs={g:np.flatnonzero(groups==g) for g in ug};rng=np.random.default_rng(364000+k);deltas=[]
    if len(ug)>1:
     for _ in range(500):
      ix=np.concatenate([ixs[g] for g in rng.choice(ug,len(ug),replace=True)])
      if m in ('AUROC','AUPRC') and len(np.unique(y[ix]))<2:continue
      deltas.append(gain(m,y[ix],a['p'][ix],new[ix]))
    ci=np.quantile(deltas,[.05,.95]).tolist() if deltas else None;boot.append(dict(part=part,groups=len(ug),ci90=ci,lower_positive=ci is not None and ci[0]>0))
  positive=all(p['gain']>0 for p in parts);strict=positive and chosen['strict_development'] and np.mean([p['gain'] for p in parts])>threshold(m) and all(p['lower_positive'] for p in boot)
  experimental=positive and np.mean([p['gain'] for p in parts])>0
  row=dict(**chosen,confirmation=parts,bootstrap=boot,strict_accepted=bool(strict),experimental_eligible=bool(experimental),official_score=None);records.append(row);save(OUT/'confirmation.json',dict(status='IN_PROGRESS',records=records));emit('confirmation',**row)
  if strict or experimental:predict(d,graphs,t,'final',chosen['mode'])
 save(OUT/'confirmation.json',dict(status='COMPLETE',signature=signature(),records=records,strict_tasks=[r['task'] for r in records if r['strict_accepted']],experimental_tasks=[r['task'] for r in records if r['experimental_eligible'] and not r['strict_accepted']]))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['graphs','smoke','pretrain','dev','select','confirm']);ap.add_argument('--part',default='fold0');ap.add_argument('--seed',type=int,default=SEEDS[0]);ap.add_argument('--modes',nargs='+',default=list(MODES));a=ap.parse_args();OUT.mkdir(exist_ok=True);d=load()
 if a.stage=='select':selection(d);return
 graphs=prepare_graphs(d)
 if a.stage=='graphs':return
 if a.stage=='smoke':smoke(d,graphs)
 elif a.stage=='pretrain':freeze_protocol();pretrain(d,graphs,a.part,a.seed)
 elif a.stage=='dev':train_dev(d,graphs,a.modes)
 elif a.stage=='confirm':confirm(d,graphs)
if __name__=='__main__':main()
