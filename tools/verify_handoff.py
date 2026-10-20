"""Standard-library handoff verification, source integrity and release checksums."""
from pathlib import Path
import argparse,ast,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifacts-dir',type=Path);p.add_argument('--check-restored',action='store_true');a=p.parse_args()
 for file in ROOT.glob('admet_*.py'):ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file))
 status=json.loads((ROOT/'CURRENT_STATUS.json').read_text(encoding='utf-8'))
 checks=[]
 for version in (61,62,63,65):
  meta=json.loads((ROOT/f'admet_model_outputs_v{version}/release.json').read_text(encoding='utf-8'))
  replay=json.loads((ROOT/f'admet_model_outputs_v{version}/model_replay.json').read_text(encoding='utf-8'))
  file=next((ROOT/f'admet_submissions/v{version}').glob('*.csv'))
  assert sha(file)==meta['sha256']==status['releases'][str(version)]['sha256']
  assert replay['status']=='PASS'
  subprocess.run([sys.executable,str(ROOT/'admet_release_local_verify.py'),str(version)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
  checks.append({'version':version,'csv_hash':'PASS','fusion':'PASS','recorded_remote_replay':'PASS'})
 remote=json.loads((ROOT/'docs/remote_snapshot_manifest.json').read_text(encoding='utf-8'))
 n=0
 for item in remote['files']:
  if item['path'].startswith('admet_') and '/' not in item['path']:
   file=ROOT/item['path']
   if file.exists():assert sha(file)==item['sha256'],item['path'];n+=1
 if a.artifacts_dir:
  manifest=json.loads((ROOT/'docs/artifact_manifest.json').read_text(encoding='utf-8'))
  for item in manifest['archives']:
   file=a.artifacts_dir/item['name'];assert file.stat().st_size==item['bytes'] and sha(file)==item['sha256'],str(file)
   if a.check_restored:
    members=json.loads((ROOT/'docs/artifact-manifests'/item['name'].replace('.tar.gz','.manifest.json')).read_text(encoding='utf-8'))
    for member in members:
     path=ROOT/member['path'];assert path.resolve().is_relative_to(ROOT)
     assert path.is_file() and sha(path)==member['sha256'],str(path)
  checks.append({'archives':'PASS','restored':a.check_restored})
 print(json.dumps({'status':'PASS','exact_remote_source_files':n,'checks':checks,'official_score':None},indent=2))
if __name__=='__main__':main()
