"""Persistent remote stage-one runner; no SSH secrets or local training."""
import json, os, time, traceback
from pathlib import Path
import v364_graph_transformer as m

ROOT=Path(__file__).resolve().parent
def status(stage,**extra):
 m.save(ROOT/'run_status.json',dict(stage=stage,pid=os.getpid(),at_unix=time.time(),**extra))
 print(json.dumps(dict(stage=stage,**extra)),flush=True)
def main():
 start=time.monotonic()
 try:
  m.setup();d=m.load();graphs=m.prepare_graphs(d)
  smoke=json.loads((m.OUT/'smoke.json').read_text());assert smoke['status']=='PASS' and smoke['signature']==m.signature()
  m.freeze_protocol();status('DEVELOPMENT',signature=m.signature())
  m.train_dev(d,graphs,m.MODES)
  status('FREEZE_SELECTION');m.selection(d)
  status('CONFIRM_AND_FINAL');m.confirm(d,graphs)
  status('TRAINING_COMPLETE',seconds=time.monotonic()-start,signature=m.signature())
 except Exception:
  status('FAILED',seconds=time.monotonic()-start,traceback=traceback.format_exc());raise
if __name__=='__main__':main()
