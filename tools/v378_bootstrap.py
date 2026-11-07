"""Prepare next frozen experiment and queue behind completed v377 evidence."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,tarfile,time,os
ROOT=Path('/root/autodl-tmp/regression9_v378')


def main():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'launch.json').exists(),'Do not duplicate job'
 archive=ROOT.parent/'v378_neural_delta.tar.gz'
 assert hashlib.sha256(archive.read_bytes()).hexdigest()=='28d9d538e78e43159f086ae70002a1b28d051fa4601e0e23be4755749206360e'
 with tarfile.open(archive) as tf:
  for member in tf.getmembers():
   dest=(ROOT/member.name).resolve();assert member.isfile() and dest.is_relative_to(ROOT)
   dest.parent.mkdir(parents=True,exist_ok=True);data=tf.extractfile(member).read()
   if dest.exists():assert dest.read_bytes()==data
   else:dest.write_bytes(data)
 prev=ROOT.parent/'regression9_v377'
 for name in ('controls','design'):
  if not (ROOT/name).exists():shutil.copytree(prev/name,ROOT/name)
 for name in ('official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv','v374_baseline.csv','data_manifest.json'):
  if not (ROOT/name).exists():shutil.copyfile(prev/name,ROOT/name)
 manifest=json.loads((ROOT/'data_manifest.json').read_text())
 for name,key in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256')):assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest[key]
 (ROOT/'queue_status.json').write_text(json.dumps(dict(status='QUEUED_AFTER_V377',pid=os.getpid()))+'\n')
 while True:
  stage=json.loads((prev/'run_status.json').read_text())['stage'];assert stage!='FAILED','v377 failed, inspect before continuing'
  if stage=='EVIDENCE_READY':break
  time.sleep(20)
 with (ROOT/'training.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,'-u','v378_run_experts.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result=dict(status='STARTED',pid=proc.pid,root=str(ROOT));(ROOT/'launch.json').write_text(json.dumps(result)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
