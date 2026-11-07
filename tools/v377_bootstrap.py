"""Extract immutable input delta, inherit hashed arrays, launch remote runner."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
ROOT=Path('/root/autodl-tmp/regression9_v377')


def main():
 ROOT.mkdir(exist_ok=True)
 assert not (ROOT/'launch.json').exists(),'Do not launch duplicate runner'
 archive=ROOT.parent/'v377_regression9_delta.tar.gz'
 assert hashlib.sha256(archive.read_bytes()).hexdigest()=='20fad5266e55597f7e29e4a9c320651e4090e5f5aa430b9e21b3fe917fbd4ab7'
 with tarfile.open(archive) as tf:
  for member in tf.getmembers():
   dest=(ROOT/member.name).resolve();assert member.isfile() and dest.is_relative_to(ROOT)
   dest.parent.mkdir(parents=True,exist_ok=True);payload=tf.extractfile(member).read()
   if dest.exists():assert dest.read_bytes()==payload
   else:dest.write_bytes(payload)
 for name in ('official_rows.npz','expert_features.npz','stereo_keys.npz'):
  dest=ROOT/name
  if not dest.exists():shutil.copyfile(ROOT.parent/'mae_experts_v374'/name,dest)
 manifest=json.loads((ROOT/'data_manifest.json').read_text())
 for name,key in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256')):assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest[key]
 for r in manifest['controls']:
  for name in ('control','design'):assert hashlib.sha256((ROOT/r[name]).read_bytes()).hexdigest()==r[name+'_sha256']
 with (ROOT/'training.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,'-u','v377_run_experts.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result=dict(status='STARTED',pid=proc.pid,root=str(ROOT));(ROOT/'launch.json').write_text(json.dumps(result)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
