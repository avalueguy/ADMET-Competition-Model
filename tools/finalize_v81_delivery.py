"""Verify the v81 archive and place the upload copy."""
from pathlib import Path,PurePosixPath
import sys,json,tarfile,hashlib,csv,io,shutil
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from verify_submission import verify
from repack_upload_template import field_spans
PREFIX='v81_multitask_residual_20261024'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def main():
 meta=load(ROOT/'releases'/(PREFIX+'.archive.json'));archive=ROOT/meta['path'];assert sha(archive)==meta['sha256'] and archive.stat().st_size==meta['bytes']
 manifest_path=ROOT/'releases'/(PREFIX+'.manifest.json');manifest=load(manifest_path);expected={r['path']:r for r in manifest['files']};seen=set()
 with tarfile.open(archive,'r|gz') as tf:
  for m in tf:
   rel=PurePosixPath(m.name);assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts and m.name not in seen;seen.add(m.name);raw=tf.extractfile(m).read()
   if m.name==manifest_path.relative_to(ROOT).as_posix():assert raw==manifest_path.read_bytes();continue
   r=expected[m.name];assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
 assert seen==set(expected)|{manifest_path.relative_to(ROOT).as_posix()}
 out=ROOT/'admet_model_outputs_v81_final';release=load(out/'release.json');audit=load(out/'model_verification.json');assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM' and audit['status']=='PASS';assert sha(out/'model_verification.json')==release['verification_sha256']
 file=ROOT/release['file'];template=ROOT/'admet_official/submission_template.csv';anchor=ROOT/'admet_submissions/v80_cyp2c9_residual/我从山北来带着兰花草-小分子药物预测技术创新赛-20261023.csv';assert sha(file)==release['sha256'] and sha(anchor)==release['anchor_sha256']
 actual=rows(file);old=rows(anchor);changed=Counter()
 for a,o in zip(actual[1:],old[1:]):
  assert a[:4]+a[5:]==o[:4]+o[5:]
  if a[4]!=o[4]:changed[a[0]]+=1
  if a[0] not in release['changed_tasks']:assert a==o
 assert dict(changed)==release['changed_tasks']
 fmt=verify(file,template);raw=file.read_bytes();ref=template.read_bytes();s=field_spans(raw);t=field_spans(ref);chunks=[];cur=0
 for i in range(1,len(actual)):
  lo,hi=s[i][4];a,b=t[i][4];chunks.extend((raw[cur:lo],ref[a:b]));cur=hi
 chunks.append(raw[cur:]);assert b''.join(chunks)==ref
 dest=Path('D:/bili_download/v81_multitask_residual');dest.mkdir(parents=True,exist_ok=True);upload=dest/file.name
 if upload.exists():assert upload.read_bytes()==raw
 else:shutil.copy2(file,upload)
 shutil.copy2(file.with_suffix('.verification.json'),upload.with_suffix('.verification.json'));shutil.copy2(ROOT/'docs/ITERATION_V81_20261024.md',dest/'v81_说明.md');assert sha(upload)==release['sha256']
 result=dict(status='LOCAL_DELIVERY_COMPLETE_AWAITING_PLATFORM',version='v81',csv=release['file'],upload_copy=str(upload),sha256=sha(file),changed_tasks=dict(changed),unchanged_tasks=18,rows=16379,columns=6,tasks=22,model_reload='PASS',models=9,nonprediction_bytes_exact=True,archive=meta,all_archive_member_hashes_verified=True,official_score=None,anchor='v80',anchor_official_score_user_reported=50.647,test_labels_used=False,external_data=False,automatic_upload=False);save(ROOT/'docs/v81_delivery.json',result);save(dest/'v81_delivery.json',result)
 state=load(ROOT/'CURRENT_STATUS.json');state.update(handoff_date='2026-10-24',latest_completed_model_branch='v81',latest_completed_model_branch_is_proven_best=False,recommended_submission_version='v81',best_official_version='v80',best_official_version_user_reported='v80',best_official_score_user_reported=50.647,active_experiment='v81',active_experiment_status=result['status'],new_submission_created=True,official_score=None,priority_next_step='Submit v81 and record the exact platform score.')
 state['active_iteration']=dict(version='v81',status=result['status'],csv=release['file'],changed_tasks=dict(changed),independent_model_verification='PASS',archive_verified=True,running_training_processes=0,official_score=None,local_delivery_report='docs/v81_delivery.json',github_push=False,server_shutdown_requested=False)
 state['releases']['81']=dict(csv=release['file'],sha256=sha(file),status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',official_score=None,baseline='v80',changed_tasks=dict(changed),unchanged_tasks=18,model_verification='PASS',archive=meta,desktop_upload_copy=str(upload));save(ROOT/'CURRENT_STATUS.json',state)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
