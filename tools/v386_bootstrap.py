"""Safely unpack frozen delta, link unchanged v385 inputs, smoke, launch once."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile
ROOT=Path('/root/autodl-tmp/regression9_v386')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads(sys.argv[1]);ROOT.mkdir(exist_ok=True);assert not(ROOT/'launch.json').exists();prev=ROOT.parent/'regression9_v385';archive=ROOT.parent/Path(r['archive']).name;assert sha(archive)==r['sha256']
 with tarfile.open(archive)as tf:
  for member in tf.getmembers():
   dest=(ROOT/member.name).resolve();assert member.isfile()and dest.is_relative_to(ROOT.resolve());dest.parent.mkdir(parents=True,exist_ok=True);data=tf.extractfile(member).read()
   if dest.exists():assert dest.read_bytes()==data
   else:dest.write_bytes(data)
 manifest=json.loads((ROOT/'data_manifest.json').read_text());names=['official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv']+[r['design']for r in manifest['controls']]+[r['control']for r in manifest['controls']if r['task']!='ppbr_az']
 for n in names:
  src,dest=prev/n,ROOT/n;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(src)==sha(dest)
  else:os.link(src,dest)
 for n,k in(('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256'),('v385_baseline.csv','baseline_csv_sha256')):assert sha(ROOT/n)==manifest[k]
 for r in manifest['controls']:
  for k in('control','design'):assert sha(ROOT/r[k])==r[k+'_sha256']
 subprocess.run([sys.executable,'-u','v386_smoke.py'],cwd=ROOT,check=True)
 with(ROOT/'training.log').open('w')as log:proc=subprocess.Popen([sys.executable,'-u','v386_run_experts.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 value=dict(status='STARTED',pid=proc.pid,root=str(ROOT),workers=2);(ROOT/'launch.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
