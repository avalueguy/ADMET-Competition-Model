"""Independent numerical and checkpoint audit; never fits models or selects fallbacks."""
from pathlib import Path
import hashlib, json, math, platform
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from scipy.special import expit
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import v369_graph_transformer as m

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def manual_encoder(net,b):
 """Separate attention implementation, checked against the optimized SDPA kernel."""
 x=torch.zeros((*b['atom'].shape[:2],m.CONFIG['width']),device=b['atom'].device)
 for i,e in enumerate(net.atom):x=x+e(b['atom'][:,:,i])
 bias=(net.distance(b['distance'])+net.bond(b['bond'])).permute(0,3,1,2)
 for block in net.blocks:
  bs,n,w=x.shape;v=block.norm1(x);qkv=block.qkv(v).view(bs,n,3,block.h,w//block.h).permute(2,0,3,1,4)
  q,k,v=qkv[0],qkv[1],qkv[2]
  logits=q@k.transpose(-2,-1)/math.sqrt(w//block.h)+bias
  logits=logits.masked_fill(~b['valid'][:,None,None,:],-torch.inf)
  attn=torch.softmax(logits,dim=-1)@v
  x=x+block.proj(attn.transpose(1,2).contiguous().view(bs,n,w));v=block.norm2(x)
  x=x+block.ff(v)+block.adapter(v)
 x=net.norm(x);valid=b['valid'].clone();valid[:,0]=False
 mean=(x*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True).clamp(min=1)
 return torch.cat((x[:,0],mean),dim=1)
def score(metric,y,p):
 if metric=='MAE':return float(mean_absolute_error(y,p))
 if metric=='Spearman':return float(spearmanr(y,p).statistic)
 if metric=='AUROC':return float(roc_auc_score(y,p))
 return float(average_precision_score(y,p))
def fusion(metric,a,b,w):
 if metric=='MAE':return a*(1-w)+b*w
 return (1-w)*rankdata(a,method='average')/len(a)+w*rankdata(b,method='average')/len(b)
def gain(metric,y,a,b):
 old,new=score(metric,y,a),score(metric,y,b)
 return (old-new)/old if metric=='MAE' else new-old
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 d=m.load();graphs=m.prepare_graphs(d);signature=m.signature()
 protocol=json.loads((m.OUT/'protocol.json').read_text());frozen=json.loads((m.OUT/'frozen_selection.json').read_text());confirmation=json.loads((m.OUT/'confirmation.json').read_text())
 assert protocol['signature']==frozen['signature']==confirmation['signature']==signature
 assert confirmation['status']=='COMPLETE'
 # Recompute the unique development winner with an independently written score/fusion.
 independently_selected=[]
 for task in m.TASKS:
  metric=d['metrics'][d['tasks'].index(task)];choices=[]
  for mode in m.MODES:
   for weight in (.2,.45):
    values=[]
    for part in ('fold0','fold1'):
     z=np.load(m.OUT/'tasks'/task/part/mode/'predictions.npz');a=m.ctrl(task,part);assert str(z['signature'])==signature
     np.testing.assert_array_equal(z['query'],a['query']);np.testing.assert_array_equal(z['fit'],a['fit'])
     values.append(gain(metric,d['y'][a['query']],a['p'],fusion(metric,a['p'],z['p'],weight)))
    choices.append((min(values),-weight,-m.MODES.index(mode),mode,weight))
  best=max(choices)
  if best[0]>0:independently_selected.append((task,best[3],best[4]))
 assert independently_selected==[(r['task'],r['mode'],r['weight']) for r in frozen['selected']]
 records=[];max_manual=max_replay=max_cpu=0.
 for row in confirmation['records']:
  task=row['task'];metric=row['metric'];mode=row['mode'];parts=[]
  approved=row['strict_accepted'] or row['experimental_eligible']
  for part in ('fold0','fold1','fold2','valid')+(('final',) if approved else ()):
   folder=m.OUT/'tasks'/task/part/mode;a=m.ctrl(task,part);fit,q=a['fit'],a['query'];z=np.load(folder/'predictions.npz')
   np.testing.assert_array_equal(z['query'],q);np.testing.assert_array_equal(z['fit'],fit);assert str(z['signature'])==signature
   assert not np.intersect1d(fit,q).size
   if part!='final':
    assert not np.intersect1d(d['rg'][fit],d['rg'][q]).size
    assert not np.intersect1d(d['canonical'][d['mi'][fit]],d['canonical'][d['mi'][q]]).size
   # Perturbing query labels must not affect normalization or target transforms.
   changed={**d,'y':d['y'].copy()};changed['y'][q]=np.arange(len(q))+98765.
   desc,ds=m.normalize_desc(d,fit);desc2,ds2=m.normalize_desc(changed,fit)
   np.testing.assert_array_equal(desc,desc2)
   yy,ys=m.target(d['y'][fit],metric);yy2,ys2=m.target(changed['y'][fit],metric);np.testing.assert_array_equal(yy,yy2);assert ys==ys2
   sample=q[np.unique(np.linspace(0,len(q)-1,min(16,len(q)),dtype=int))];members=[];checks=[]
   for n,seed in enumerate(m.SEEDS):
    pth=folder/f'model_{seed}.pt';s=torch.load(pth,map_location='cpu',weights_only=False)
    assert s['signature']==signature;np.testing.assert_array_equal(s['fit'],fit);np.testing.assert_array_equal(s['query'],q)
    for k in ds:np.testing.assert_array_equal(s['descriptor'][k],ds[k])
    assert s['target']==ys
    net=m.Predictor(desc.shape[1]).eval();net.load_state_dict(s['weights'])
    b=m.batch(graphs,d['mi'][sample],device='cpu');xx=torch.as_tensor(desc[d['mi'][sample]])
    with torch.inference_mode():
     reference=net.head(torch.cat([manual_encoder(net.encoder,b),net.desc(xx)],1)).squeeze(1).numpy()
     optimized=net(b,xx).numpy();manual_error=float(np.max(np.abs(reference-optimized)));assert manual_error<.00005,manual_error
    max_manual=max(max_manual,manual_error)
    # CPU independently recomputes representative examples from stored model weights.
    indices=np.searchsorted(q,sample);assert np.all(q[indices]==sample)
    cpu_error=float(np.max(np.abs(optimized-z['raw_members'][n,indices])));assert cpu_error<.0001,(task,part,cpu_error)
    max_cpu=max(max_cpu,cpu_error)
    # All examples replayed on the exact training GPU; no optimizer is constructed.
    net=net.cuda();allraw=m.infer(net,d,graphs,q,torch.as_tensor(desc,device='cuda'))
    replay_error=float(np.max(np.abs(allraw-z['raw_members'][n])));assert replay_error<.00001,replay_error;max_replay=max(max_replay,replay_error)
    order=np.arange(len(sample))[::-1];reversed_raw=m.infer(net,d,graphs,sample[order],torch.as_tensor(desc,device='cuda'),chunk=3)[::-1]
    assert np.max(np.abs(reversed_raw-optimized))<.0001
    members.append(allraw);checks.append(dict(seed=seed,checkpoint_sha256=sha(pth),manual_attention_error=manual_error,cpu_error=cpu_error,full_replay_error=replay_error))
    if mode!='scratch':
     pre=torch.load(m.OUT/'encoders'/part/str(seed)/'pretrained.pt',map_location='cpu',weights_only=False);expected=m.global_fit(d,part)
     np.testing.assert_array_equal(pre['fit_rows'],expected);np.testing.assert_array_equal(pre['fit_molecules'],np.unique(d['mi'][expected]));assert pre['signature']==signature
    del net;torch.cuda.empty_cache()
   raw=np.mean(members,axis=0);expected=expit(raw) if metric in ('AUROC','AUPRC') else raw*ys['scale']+ys['center']
   if task=='ppbr_az':expected=np.clip(expected,0,100)
   np.testing.assert_allclose(expected,z['p'],rtol=1e-5,atol=1e-6)
   blended=fusion(metric,a['p'],z['p'],row['weight']);assert np.isfinite(blended).all()
   if part!='final':
    report=next(r for r in row['confirmation'] if r['part']==part);actual_gain=gain(metric,d['y'][q],a['p'],blended)
    assert abs(actual_gain-report['gain'])<1e-12
    if part in ('fold2','valid'):
     groups=d['rg'][q];unique=np.unique(groups);indices={g:np.flatnonzero(groups==g) for g in unique};rng=np.random.default_rng(364000+d['tasks'].index(task));deltas=[]
     if len(unique)>1:
      for _ in range(500):
       ix=np.concatenate([indices[g] for g in rng.choice(unique,len(unique),replace=True)])
       if metric in ('AUROC','AUPRC') and np.unique(d['y'][q][ix]).size<2:continue
       deltas.append(gain(metric,d['y'][q][ix],a['p'][ix],blended[ix]))
     old_ci=next(r for r in row['bootstrap'] if r['part']==part)
     if deltas:
      ci=np.quantile(deltas,[.05,.95]);np.testing.assert_allclose(ci,old_ci['ci90'],atol=1e-12,rtol=0);assert bool(ci[0]>0)==old_ci['lower_positive']
     else:assert old_ci['ci90'] is None and not old_ci['lower_positive']
   parts.append(dict(part=part,fit=len(fit),query=len(q),query_label_perturbation_invariant=True,checks=checks))
  positive=all(r['gain']>0 for r in row['confirmation']);practical=.012 if metric=='MAE' else .008 if metric=='Spearman' else .003
  strict=positive and row['strict_development'] and np.mean([r['gain'] for r in row['confirmation']])>practical and all(r['lower_positive'] for r in row['bootstrap'])
  assert bool(strict)==row['strict_accepted'];assert bool(positive)==row['experimental_eligible']
  records.append(dict(task=task,strict_accepted=row['strict_accepted'],experimental_eligible=row['experimental_eligible'],parts=parts))
  print(json.dumps(dict(stage='verified',task=task,parts=len(parts))),flush=True)
 result=dict(status='PASS',signature=signature,source_sha256=sha(__file__),records=records,manual_attention_max_error=max_manual,cpu_max_error=max_cpu,full_replay_max_error=max_replay,test_labels_used=False,verification_only_no_training=True,official_score=None,baseline_oof_limitation='Partial historical proxy, retained explicitly')
 m.save(m.OUT/'independent_verification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
