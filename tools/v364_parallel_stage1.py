"""Three bounded processes on the authorized GPU, with checkpoint and encoder locks."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing, fcntl, json, os, time, traceback
import v364_graph_transformer as m
ROOT=Path(__file__).resolve().parent
DATA=GRAPHS=None
def initializer():
 global DATA,GRAPHS
 m.setup();DATA=m.load();GRAPHS=m.prepare_graphs(DATA)
 original=m.pretrain
 def locked_pretrain(d,graphs,part,seed):
  p=m.OUT/'encoders'/part/str(seed);p.mkdir(parents=True,exist_ok=True)
  with open(p/'pretrain.lock','a') as f:
   fcntl.flock(f,fcntl.LOCK_EX)
   return original(d,graphs,part,seed)
 m.pretrain=locked_pretrain
def work(job):
 part,mode,task=job;folder=m.OUT/'tasks'/task/part/mode;folder.mkdir(parents=True,exist_ok=True)
 with open(folder/'training.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);m.predict(DATA,GRAPHS,task,part,mode)
 return dict(part=part,mode=mode,task=task)
def status(stage,**extra):
 m.save(ROOT/'run_status.json',dict(stage=stage,pid=os.getpid(),at_unix=time.time(),workers=3,**extra));print(json.dumps(dict(stage=stage,**extra)),flush=True)
def main():
 start=time.monotonic()
 try:
  smoke=json.loads((m.OUT/'smoke.json').read_text());assert smoke['status']=='PASS' and smoke['signature']==m.signature();m.freeze_protocol()
  status('DEVELOPMENT_PARALLEL',signature=m.signature(),resume_existing_checkpoints=True,model_or_budget_changes=False)
  jobs=[(p,mode,t) for p in ('fold0','fold1') for mode in m.MODES for t in m.TASKS]
  with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),initializer=initializer) as pool:
   for i,r in enumerate(pool.map(work,jobs),1):status('DEVELOPMENT_PARALLEL',completed_jobs=i,total_jobs=len(jobs),latest=r)
  m.setup();d=m.load();graphs=m.prepare_graphs(d)
  status('FREEZE_SELECTION');m.selection(d)
  status('CONFIRM_AND_FINAL');m.confirm(d,graphs)
  status('TRAINING_COMPLETE',seconds=time.monotonic()-start,signature=m.signature())
 except Exception:
  status('FAILED',seconds=time.monotonic()-start,traceback=traceback.format_exc());raise
if __name__=='__main__':main()
