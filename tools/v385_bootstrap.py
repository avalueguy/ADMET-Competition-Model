"""Launch frozen warp experiment after remote validation and fit smoke."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile
ROOT=Path('/root/autodl-tmp/regression9_v385')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 receipt=json.loads(sys.argv[1]);ROOT.mkdir(exist_ok=True);assert not(ROOT/'launch.json').exists();prev=ROOT.parent/'regression9_v384'
 archive=ROOT.parent/Path(receipt['archive']).name;assert sha(archive)==receipt['sha256']
 with tarfile.open(archive)as tf:
  for member in tf.getmembers():
   dest=(ROOT/member.name).resolve();assert member.isfile()and dest.is_relative_to(ROOT.resolve());dest.parent.mkdir(parents=True,exist_ok=True);data=tf.extractfile(member).read()
   if dest.exists():assert dest.read_bytes()==data
   else:dest.write_bytes(data)
 manifest=json.loads((prev/'data_manifest.json').read_text());names=['official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv','v382_baseline.csv','data_manifest.json']+[r[k]for r in manifest['controls']for k in('control','design')]
 for n in names:
  src,dest=prev/n,ROOT/n;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(src)==sha(dest)
  else:os.link(src,dest)
 for r in manifest['controls']:
  for k in('control','design'):assert sha(ROOT/r[k])==r[k+'_sha256']
 subprocess.run([sys.executable,'-u','v385_smoke.py'],cwd=ROOT,check=True)
 with(ROOT/'training.log').open('w')as log:proc=subprocess.Popen([sys.executable,'-u','v385_run_experts.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 value=dict(status='STARTED',pid=proc.pid,root=str(ROOT),workers=2,input_receipt=receipt);(ROOT/'launch.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
