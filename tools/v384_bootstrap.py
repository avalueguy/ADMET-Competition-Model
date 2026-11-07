"""Validate frozen inputs, smoke and launch once on the designated host."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile
ROOT=Path('/root/autodl-tmp/regression9_v384')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 receipt=json.loads(sys.argv[1]);ROOT.mkdir(exist_ok=True);assert not(ROOT/'launch.json').exists()
 prev=ROOT.parent/'regression9_v377';archive=ROOT.parent/Path(receipt['archive']).name;assert sha(archive)==receipt['sha256']
 with tarfile.open(archive)as tf:
  for member in tf.getmembers():
   dest=(ROOT/member.name).resolve();assert member.isfile()and dest.is_relative_to(ROOT.resolve());dest.parent.mkdir(parents=True,exist_ok=True);data=tf.extractfile(member).read()
   if dest.exists():assert dest.read_bytes()==data
   else:dest.write_bytes(data)
 manifest=json.loads((ROOT/'data_manifest.json').read_text())
 names=['official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv']
 names += [r['design']for r in manifest['controls']]+[r['control']for r in manifest['controls']if r['task']not in('lipophilicity_astrazeneca','ppbr_az')]
 for n in names:
  src,dest=prev/n,ROOT/n;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(src)==sha(dest)
  else:os.link(src,dest)
 for n,k in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256'),('v382_baseline.csv','v382_baseline_sha256')):assert sha(ROOT/n)==manifest[k]
 for r in manifest['controls']:
  for key in('control','design'):assert sha(ROOT/r[key])==r[key+'_sha256']
 subprocess.run([sys.executable,'-u','v384_smoke.py'],cwd=ROOT,check=True)
 with(ROOT/'training.log').open('w')as log:proc=subprocess.Popen([sys.executable,'-u','v384_run_experts.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 value=dict(status='STARTED',pid=proc.pid,root=str(ROOT),workers=2,input_receipt=receipt)
 (ROOT/'launch.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
