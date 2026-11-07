"""Run bounded task experts, freeze on development, confirm then train final."""
from pathlib import Path
import os,json,time,traceback,subprocess,sys
import numpy as np
import v371_mae_experts as m

ROOT=Path(__file__).resolve().parent


def status(stage,**x):
 m.save(ROOT/'run_status.json',dict(stage=stage,pid=os.getpid(),at_unix=time.time(),**x));m.emit(stage,**x)
def gain(y,a,p):return float((np.abs(y-a).mean()-np.abs(y-p).mean())/np.abs(y-a).mean())
def report(d,t,part,c,w):
 a=m.ctrl(t,part);p=m.predict(d,t,part,c);y=d['y'][a['query']];new=(1-w)*a['p']+w*p
 return dict(part=part,baseline=float(np.abs(y-a['p']).mean()),expert=float(np.abs(y-p).mean()),candidate=float(np.abs(y-new).mean()),gain=gain(y,a['p'],new))
def bootstrap(d,task,part,c,w):
 a=m.ctrl(task,part);q=a['query'];p=(1-w)*a['p']+w*m.predict(d,task,part,c);y=d['y'][q];g=d['rg'][q];unique=np.unique(g);indices=[np.flatnonzero(g==v) for v in unique];rng=np.random.default_rng(371000+d['tasks'].index(task));values=[]
 if len(unique)>1:
  for _ in range(500):
   ix=np.concatenate([indices[j] for j in rng.integers(len(unique),size=len(unique))]);values.append(gain(y[ix],a['p'][ix],p[ix]))
 ci=np.quantile(values,[.05,.95]).tolist() if values else None
 return dict(part=part,groups=len(unique),ci90=ci,lower_positive=bool(ci and ci[0]>0))
def smoke(d):
 m.setup();a=m.design(d,m.TASKS[0],'fold0');sub={k:v.copy() for k,v in a.items()};sub.update(fit=a['fit'][:70],query=a['query'][:9],x=a['x'][:70],qx=a['qx'][:9]);k,kq,_=m.kernels(d,sub)
 np.testing.assert_allclose(k,k.T,atol=1e-6);assert np.linalg.eigvalsh(k.astype(float)).min()>-1e-4
 from xgboost import XGBRegressor
 x,qx,_=m.tree_design(d,sub);net=XGBRegressor(objective='reg:quantileerror',quantile_alpha=.5,n_estimators=8,max_depth=2,device='cuda',tree_method='hist',n_jobs=4,random_state=371)
 net.fit(x,d['y'][sub['fit']],verbose=False);assert np.isfinite(net.predict(qx)).all()
 yy,c,s=m.target(d['y'][sub['fit']]);net2=m.SVR(kernel='precomputed',C=3.,epsilon=.03).fit(k,yy);assert net2.fit_status_==0
 assert m.weighted_median(np.array([[0.,2.,9.]]),np.array([[1.,2.,1.]])).item()==2.
 m.save(m.OUT/'smoke.json',dict(status='PASS',signature=m.signature(),cuda_device=m.torch.cuda.get_device_name(),quantile_gpu_fit=True,kernel_psd=True,svr_converged=True,weighted_median_checked=True,query_label_used=False))
def main():
 start=time.monotonic();m.OUT.mkdir(exist_ok=True)
 try:
  d=m.load();m.freeze();status('SMOKE',signature=m.signature());smoke(d);status('DEVELOPMENT')
  n=0
  for part in ('fold0','fold1'):
   for task in m.TASKS:
    for c in m.CONFIGS:
     m.predict(d,task,part,c);n+=1;m.emit('job_done',completed=n,total=70,task=task,part=part,config=c['name'])
  records=[];selected=[]
  for task in m.TASKS:
   choices=[]
   for c in m.CONFIGS:
    for w in m.WEIGHTS:
     parts=[report(d,task,p,c,w) for p in ('fold0','fold1')];choices.append(dict(task=task,config=c,weight=w,parts=parts,objective=min(r['gain'] for r in parts)))
   best=max(choices,key=lambda x:(x['objective'],-x['weight'],-[c['name'] for c in m.CONFIGS].index(x['config']['name'])))
   records.append(dict(task=task,choices=choices,best=best))
   if best['objective']>0:selected.append(dict(**best,strict_development=best['objective']>.012))
   m.emit('development_selection',**best)
  m.save(m.OUT/'development.json',dict(status='COMPLETE',signature=m.signature(),records=records))
  path=m.OUT/'frozen_selection.json';value=dict(signature=m.signature(),selected=selected,no_confirmation_fallback=True)
  if path.exists():assert json.loads(path.read_text())==value
  else:m.save(path,value)
  status('CONFIRMATION',selected=[s['task'] for s in selected]);confirmed=[]
  for s in selected:
   t,c,w=s['task'],s['config'],s['weight'];parts=[report(d,t,p,c,w) for p in ('fold0','fold1','fold2','valid')];boot=[bootstrap(d,t,p,c,w) for p in ('fold2','valid')]
   positive=all(r['gain']>0 for r in parts);strict=positive and s['strict_development'] and np.mean([r['gain'] for r in parts])>.012 and all(b['lower_positive'] for b in boot)
   r=dict(**s,confirmation=parts,bootstrap=boot,strict_accepted=bool(strict),experimental_eligible=bool(positive),ten_percent_all_parts=all(x['gain']>=.1 for x in parts),official_score=None);confirmed.append(r)
   m.save(m.OUT/'confirmation.json',dict(status='IN_PROGRESS',records=confirmed));m.emit('confirmation_result',**r)
   if positive:m.predict(d,t,'final',c)
  m.save(m.OUT/'confirmation.json',dict(status='COMPLETE',signature=m.signature(),records=confirmed,strict_tasks=[r['task'] for r in confirmed if r['strict_accepted']],experimental_tasks=[r['task'] for r in confirmed if r['experimental_eligible'] and not r['strict_accepted']]))
  status('VERIFYING',seconds=time.monotonic()-start)
  subprocess.run([sys.executable,'-u','v371_verify_experts.py'],cwd=ROOT,check=True)
  status('EVIDENCE_READY',seconds=time.monotonic()-start,signature=m.signature())
 except Exception:status('FAILED',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
