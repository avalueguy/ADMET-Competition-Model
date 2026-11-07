"""Complete bounded nine-task learning curves, freeze choice, then confirm."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback
import numpy as np
from scipy.stats import rankdata,spearmanr
import v378_retrieval_transformer as m
ROOT=Path(__file__).resolve().parent


def status(stage,**kw):m.save(ROOT/'run_status.json',dict(stage=stage,pid=os.getpid(),at_unix=time.time(),**kw));m.emit(stage,**kw)
def score(d,t,p,c,w):
 a=m.ctrl(t,p);y=d['y'][a['query']];expert=m.predict(d,t,p,c);new=(1-w)*a['p']+w*expert
 old=float(np.abs(y-a['p']).mean());mae=float(np.abs(y-new).mean())
 value=dict(part=p,baseline=old,candidate=mae,expert=float(np.abs(y-expert).mean()),gain=(old-mae)/old)
 if t in m.TASKS[5:]:
  official=(1-w)*rankdata(a['score_p'])/len(y)+w*rankdata(expert)/len(y)
  oldr=float(spearmanr(y,a['score_p']).statistic);newr=float(spearmanr(y,official).statistic)
  value.update(baseline_spearman=oldr,candidate_spearman=newr,rank_gain=newr-oldr)
 return value


def boot(d,t,part,c,w):
 a=m.ctrl(t,part);y=d['y'][a['query']];expert=m.predict(d,t,part,c);new=(1-w)*a['p']+w*expert
 group=d['rg'][a['query']];u=np.unique(group);chunks=[np.flatnonzero(group==g) for g in u]
 rng=np.random.default_rng(378000+d['tasks'].index(t));values=[]
 for _ in range(500):
  ix=np.concatenate([chunks[j] for j in rng.integers(len(u),size=len(u))]);old=np.abs(y[ix]-a['p'][ix]).mean()
  values.append(float((old-np.abs(y[ix]-new[ix]).mean())/old))
 ci=np.quantile(values,[.05,.95]).tolist()
 return dict(part=part,groups=len(u),ci90=ci,lower_positive=ci[0]>0)


def main():
 started=time.monotonic();m.OUT.mkdir(exist_ok=True)
 try:
  d=m.load();m.setup();m.freeze();status('DEVELOPMENT',signature=m.signature());done=0
  for part in ('fold0','fold1'):
   for t in m.TASKS:
    for b in m.BASES:
     m.train_curve(d,t,part,b);done+=1;m.emit('curve_job_done',completed=done,total=36)
  records=[];selected=[]
  for t in m.TASKS:
   choices=[]
   for c in m.CONFIGS:
    for w in m.WEIGHTS:
     parts=[score(d,t,p,c,w) for p in ('fold0','fold1')];objective=min(r['gain'] for r in parts)
     # New ranking tasks must improve the official metric, too.
     if t in m.TASKS[5:] and min(r['rank_gain'] for r in parts)<=0:objective=min(objective,-1.)
     choices.append(dict(task=t,config=c,weight=w,parts=parts,objective=objective))
   best=max(choices,key=lambda r:(r['objective'],-r['weight'],-r['config']['rounds'],-list(m.CONFIGS).index(r['config'])))
   records.append(dict(task=t,best=best,choices=choices));m.emit('development_selection',**best)
   if best['objective']>0:selected.append(best)
  m.save(m.OUT/'development.json',dict(signature=m.signature(),status='COMPLETE',records=records))
  m.save(m.OUT/'frozen_selection.json',dict(signature=m.signature(),selected=selected,no_confirmation_fallback=True))
  status('CONFIRMATION',tasks=[r['task'] for r in selected]);confirmed=[]
  for r in selected:
   t,c,w=r['task'],r['config'],r['weight'];parts=[score(d,t,p,c,w) for p in ('fold0','fold1','fold2','valid')];boots=[boot(d,t,p,c,w) for p in ('fold2','valid')]
   positive=all(p['gain']>0 for p in parts) and all(p.get('rank_gain',1)>0 for p in parts)
   strict=positive and min(p['gain'] for p in parts[:2])>.012 and np.mean([p['gain'] for p in parts])>.012 and all(b['lower_positive'] for b in boots)
   result=dict(**r,confirmation=parts,bootstrap=boots,experimental_eligible=bool(positive),strict_accepted=bool(strict));confirmed.append(result)
   m.emit('confirmation_result',**result)
   if positive:m.predict(d,t,'final',c)
  m.save(m.OUT/'confirmation.json',dict(status='COMPLETE',signature=m.signature(),records=confirmed))
  status('VERIFYING');subprocess.run([sys.executable,'-u','v378_verify_experts.py'],cwd=ROOT,check=True)
  status('EVIDENCE_READY',seconds=time.monotonic()-started,signature=m.signature())
 except Exception:status('FAILED',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
