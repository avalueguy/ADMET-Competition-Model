"""Collect compact evidence and adopted final weights, preserving rejected experiments remotely."""
import hashlib, json, tarfile
from pathlib import Path
import v364_graph_transformer as m
ROOT=Path(__file__).resolve().parent
def main():
 checked=json.loads((m.OUT/'independent_verification.json').read_text());assert checked['status']=='PASS' and checked['signature']==m.signature()
 report=json.loads((m.OUT/'confirmation.json').read_text());adopted=[r for r in report['records'] if r['strict_accepted'] or r['experimental_eligible']]
 files=[]
 for p in m.OUT.rglob('*'):
  if p.is_file() and p.suffix!='.pt':files.append(p)
 checkpoints=[]
 for row in adopted:
  for p in sorted((m.OUT/'tasks'/row['task']/'final'/row['mode']).glob('*.pt')):
   checkpoints.append(dict(remote_path=str(p),relative_path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=m.sha(p)))
 files += [ROOT/'v364_graph_transformer.py',ROOT/'v364_run_stage1.py',ROOT/'v364_parallel_stage1.py',ROOT/'v364_verify_remote.py',ROOT/'v364_collect_remote.py',ROOT/'run_status.json',ROOT/'graph_audit.json',ROOT/'training.log',ROOT/'parallel.log']
 files += [ROOT/'v364_parallel_confirmation.py',ROOT/'v364_after_training.py',ROOT/'confirmation_parallel.log',ROOT/'verification.log']
 # verification.log is active during collection and is therefore excluded from hashed artifacts.
 files=[p for p in files if p.name!='verification.log']
 entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=m.sha(p)) for p in sorted(set(files))]
 manifest=ROOT/'result_manifest.json';m.save(manifest,dict(signature=m.signature(),files=entries,all_original_checkpoints_retained_remote=True,adopted_final_checkpoints_included=False,adopted_final_checkpoints=checkpoints,source_directory=str(ROOT),model_training_gpu_remote_only=True))
 archive=ROOT/'results.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
  for p in sorted(set(files+[manifest])):tf.add(p,arcname=p.relative_to(ROOT).as_posix())
 with tarfile.open(archive) as tf:
  for row in entries:assert hashlib.sha256(tf.extractfile(row['path']).read()).hexdigest()==row['sha256']
 result=dict(status='PASS',archive=archive.name,bytes=archive.stat().st_size,sha256=m.sha(archive),adopted_tasks=[r['task'] for r in adopted],files=len(entries),signature=m.signature())
 m.save(ROOT/'result_archive.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
