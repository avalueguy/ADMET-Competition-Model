"""Bounded parallel stage two, freeze then confirm, finally verify and collect."""
from concurrent.futures import ProcessPoolExecutor
import fcntl, json, multiprocessing, os, subprocess, sys, time, traceback
from pathlib import Path
import v366_graph_transformer as m
ROOT=Path(__file__).resolve().parent
DATA=GRAPHS=None
def initializer():
 global DATA,GRAPHS
 m.setup();DATA=m.load();GRAPHS=m.prepare_graphs(DATA);original=m.base.pretrain
 def locked(d,g,p,s):
  folder=m.OUT/'encoders'/p/str(s);folder.mkdir(parents=True,exist_ok=True)
  with open(folder/'lock','a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);return original(d,g,p,s)
 m.base.pretrain=locked
def work(job):
 part,mode,task=job;folder=m.OUT/'tasks'/task/part/mode;folder.mkdir(parents=True,exist_ok=True)
 with open(folder/'lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);m.predict(DATA,GRAPHS,task,part,mode)
 return dict(part=part,mode=mode,task=task)
def status(stage,**x):
 m.save(ROOT/'run_status.json',dict(stage=stage,pid=os.getpid(),at_unix=time.time(),**x));print(json.dumps(dict(stage=stage,**x)),flush=True)
def jobs_run(jobs):
 if not jobs:return
 with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),initializer=initializer) as pool:
  for i,r in enumerate(pool.map(work,jobs),1):print(json.dumps(dict(stage='job_done',completed=i,total=len(jobs),**r)),flush=True)
def main():
 start=time.monotonic();m.OUT.mkdir(exist_ok=True)
 try:
  smoke=json.loads((m.OUT/'smoke.json').read_text());assert smoke['status']=='PASS' and smoke['signature']==m.signature();m.freeze_protocol()
  status('DEVELOPMENT',signature=m.signature());jobs_run([(p,mode,t) for p in ('fold0','fold1') for mode in m.MODES for t in m.TASKS])
  m.setup();d=m.load();g=m.prepare_graphs(d);status('FREEZE_SELECTION');m.selection(d);frozen=json.loads((m.OUT/'frozen_selection.json').read_text())
  status('CONFIRMATION');jobs_run([(p,r['mode'],r['task']) for p in ('fold2','valid') for r in frozen['selected']])
  deferred=[];original=m.base.predict
  def defer_final(d,g,t,p,mode):
   if p=='final':deferred.append((p,mode,t));return None
   return original(d,g,t,p,mode)
  m.base.predict=defer_final;m.confirm(d,g);m.base.predict=original
  status('FINAL_TRAINING',jobs=len(deferred));jobs_run(deferred);status('TRAINING_COMPLETE',seconds=time.monotonic()-start,signature=m.signature())
  for script in ('v366_verify_remote.py','v366_collect_remote.py'):subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,check=True)
  status('EVIDENCE_READY',seconds=time.monotonic()-start,signature=m.signature())
 except Exception:status('FAILED',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
