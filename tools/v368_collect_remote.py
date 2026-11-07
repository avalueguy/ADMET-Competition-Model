"""Collect stage-three evidence while retaining all full checkpoints remotely."""
from pathlib import Path
import hashlib, json, tarfile
import v368_graph_transformer as m
ROOT=Path(__file__).resolve().parent
def main():
 verified=json.loads((m.OUT/'independent_verification.json').read_text());assert verified['status']=='PASS' and verified['signature']==m.signature()
 report=json.loads((m.OUT/'confirmation.json').read_text());approved=[r for r in report['records'] if r['strict_accepted'] or r['experimental_eligible']]
 files=[p for p in m.OUT.rglob('*') if p.is_file() and p.suffix not in ('.pt',) and p.name not in ('lock',)]
 # The parent writes to training.log and run_status after collection; hash stable snapshots.
 for original,snapshot in (('training.log','completed_training_log.txt'),('run_status.json','completed_run_status.json')):
  (ROOT/snapshot).write_bytes((ROOT/original).read_bytes())
 files += [ROOT/name for name in ('v364_graph_transformer.py','v368_graph_transformer.py','v368_run_stage3.py','v368_verify_remote.py','v368_collect_remote.py','smoke.log','completed_training_log.txt','completed_run_status.json','graph_audit.json')]
 entries=[dict(path=p.relative_to(ROOT).as_posix(),sha256=m.sha(p),bytes=p.stat().st_size) for p in sorted(set(files))]
 weights=[]
 for r in approved:
  for p in sorted((m.OUT/'tasks'/r['task']/'final'/r['mode']).glob('*.pt')):weights.append(dict(remote_path=str(p),relative_path=p.relative_to(ROOT).as_posix(),sha256=m.sha(p),bytes=p.stat().st_size))
 manifest=ROOT/'result_manifest.json';m.save(manifest,dict(signature=m.signature(),files=entries,all_checkpoints_retained_remote=True,final_checkpoints=weights,training_gpu_remote_only=True))
 archive=ROOT/'results.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for p in sorted(set(files+[manifest])):tf.add(p,arcname=p.relative_to(ROOT).as_posix())
 with tarfile.open(archive) as tf:
  for r in entries:assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest()==r['sha256']
 result=dict(status='PASS',archive=archive.name,sha256=m.sha(archive),bytes=archive.stat().st_size,files=len(entries),approved_tasks=[r['task'] for r in approved],signature=m.signature())
 m.save(ROOT/'result_archive.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
