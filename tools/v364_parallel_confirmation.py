"""Complete the frozen confirmation jobs concurrently without any new selection."""
import json, multiprocessing, os, time, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import v364_graph_transformer as m
from v364_parallel_stage1 import initializer, work
ROOT=Path(__file__).resolve().parent
def status(stage,**extra):
 m.save(ROOT/'run_status.json',dict(stage=stage,pid=os.getpid(),at_unix=time.time(),workers=3,**extra));print(json.dumps(dict(stage=stage,**extra)),flush=True)
def run_jobs(jobs):
 if not jobs:return
 with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),initializer=initializer) as pool:
  for result in pool.map(work,jobs):print(json.dumps(dict(stage='frozen_job_done',**result)),flush=True)
def main():
 start=time.monotonic()
 try:
  frozen=json.loads((m.OUT/'frozen_selection.json').read_text());assert frozen['signature']==m.signature()
  assert not (m.OUT/'confirmation.json').exists(),'Do not overwrite a previous completed confirmation'
  status('CONFIRMATION_PARALLEL',signature=m.signature(),choices_frozen=True)
  jobs=[(p,r['mode'],r['task']) for p in ('fold2','valid') for r in frozen['selected']];run_jobs(jobs)
  m.setup();d=m.load();graphs=m.prepare_graphs(d);original=m.predict;deferred=[]
  def defer_final(d,graphs,task,part,mode):
   if part=='final':deferred.append((part,mode,task));return None
   return original(d,graphs,task,part,mode)
  m.predict=defer_final;m.confirm(d,graphs);m.predict=original
  status('FINAL_MODELS_PARALLEL',jobs=len(deferred));run_jobs(deferred)
  status('TRAINING_COMPLETE',seconds=time.monotonic()-start,signature=m.signature())
 except Exception:
  status('FAILED',seconds=time.monotonic()-start,traceback=traceback.format_exc());raise
if __name__=='__main__':main()
