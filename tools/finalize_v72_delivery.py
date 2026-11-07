"""Verify the local release archive and publish its local delivery record."""
from pathlib import Path
import csv, json, hashlib, sys, shutil, tarfile
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from verify_submission import verify, read
from repack_upload_template import field_spans

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    out=ROOT/'admet_model_outputs_v72_v56'
    release=json.loads((out/'release.json').read_text(encoding='utf-8'))
    assert release['status']=='CANDIDATE_READY_AWAITING_PLATFORM'
    model_audit=json.loads((out/'model_verification.json').read_text(encoding='utf-8'))
    assert model_audit['status']=='PASS' and len(model_audit['final_records'])==2
    archive_meta=json.loads((ROOT/'releases/v72_v56_20261021.archive.json').read_text(encoding='utf-8'))
    archive=ROOT/archive_meta['path'];assert sha(archive)==archive_meta['sha256']
    manifest=json.loads((ROOT/'releases/v72_v56_20261021.manifest.json').read_text(encoding='utf-8'))
    with tarfile.open(archive,'r:gz') as tf:
        for r in manifest['files']:
            data=tf.extractfile(r['path']).read()
            assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],r['path']
    candidate=ROOT/release['file'];baseline=next((ROOT/'admet_submissions/v56').glob('*.csv'))
    assert sha(candidate)==release['sha256'] and sha(baseline)==release['source_sha256']
    template=ROOT/'admet_official/submission_template.csv'
    audit=verify(candidate,template);rows,old=read(candidate),read(baseline)
    replay=json.loads((out/'fusion_replay.json').read_text(encoding='utf-8'))
    changed={}
    for task in dict.fromkeys(r[0] for r in rows[1:]):
        a=[r for r in rows[1:] if r[0]==task];b=[r for r in old[1:] if r[0]==task]
        if task not in replay:assert a==b;continue
        r=replay[task];p=np.array([float(x[4]) for x in b]);s=np.array(r['specialist'])
        expected=(1-r['weight'])*rankdata(p)/len(p)+r['weight']*rankdata(s)/len(s)
        np.testing.assert_array_equal(expected,np.array([float(x[4]) for x in a]))
        changed[task]=sum(x[4]!=y[4] for x,y in zip(a,b))
    raw=candidate.read_bytes();ref=template.read_bytes();spans=field_spans(raw);refspans=field_spans(ref)
    chunks=[];cursor=0
    for i in range(1,len(rows)):
        lo,hi=spans[i][4];a,b=refspans[i][4];chunks.extend((raw[cursor:lo],ref[a:b]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    audit.update(model_reload_status='PASS',independent_fusion_replay='PASS',nonprediction_bytes_exact=True,
        unchanged_tasks_prediction_strings_exact=True,baseline_version='v56',baseline_sha256=sha(baseline),
        baseline_official_score_user_reported=50.603,changed_tasks=changed,archive_sha256=sha(archive),
        models_backed_up_locally=True)
    save(candidate.with_suffix('.verification.json'),audit)
    delivery_dir=Path('D:/bili_download/v72_v56');delivery_dir.mkdir(parents=True,exist_ok=True)
    delivery=delivery_dir/candidate.name
    for src,dest in [(candidate,delivery),(candidate.with_suffix('.verification.json'),delivery.with_suffix('.verification.json'))]:
        if dest.exists():assert dest.read_bytes()==src.read_bytes()
        else:shutil.copy2(src,dest)
    rollback_dir=Path('D:/bili_download/v56_rollback');rollback_dir.mkdir(parents=True,exist_ok=True)
    rollback=rollback_dir/baseline.name
    if rollback.exists():assert rollback.read_bytes()==baseline.read_bytes()
    else:shutil.copy2(baseline,rollback)
    feedback_path=ROOT/'docs/platform_feedback_20261020.json'
    previous=json.loads(feedback_path.read_text(encoding='utf-8'))
    if previous.get('latest_feedback_date')!='2026-10-21':
        save(ROOT/'docs/platform_feedback_before_v72.json',previous)
    previous.update(latest_feedback_date='2026-10-21',latest_source='User direct message in current session',
        version='v56',official_score_user_reported=50.603,best_user_reported_version='v56',
        corrected_version_association='Latest user clarification associates 50.603 with v56; earlier v63 attribution is superseded.',
        v68_official_score_user_reported=49.8,v72_official_score=None,
        uploaded_byte_hash_independently_confirmed=False)
    previous['latest_update']=dict(version='v68',official_score=49.8,source='User direct message 2026-10-21')
    save(feedback_path,previous)
    state=json.loads((ROOT/'CURRENT_STATUS.json').read_text(encoding='utf-8'))
    state.update(handoff_date='2026-10-21',latest_completed_model_branch='v72',
        latest_completed_model_branch_is_proven_best=False,rollback_baseline='v56',
        best_official_version_user_reported='v56',best_official_score_user_reported=50.603,
        official_score=None,target_achieved=False,last_user_reported_score_upper_bound=None,
        automatic_platform_submission=False,local_gpu_training=False)
    state['latest_user_reported_official_score']=dict(version='v68',score=49.8,
        reported_date='2026-10-21',uploaded_byte_hash_confirmed=False,
        best_version_user_reported='v56',best_score_user_reported=50.603)
    state['active_iteration_20261021']=dict(status='LOCAL_DELIVERY_COMPLETE',baseline='v56',candidate='v72',
        new_training_host='region-9.autodl.pro:21838',training_gpu='NVIDIA A100 40GB',
        historical_v69_v70_remote_weights_not_recovered=True,github_push=False,
        running_training_processes=0,server_shutdown_requested=False)
    for value in state.values():
        if isinstance(value,dict) and value.get('aggressive_version')==69:
            value['historical_snapshot_superseded']=True
            value['later_context']='Previous server expired. Current iteration uses recovered v56 and new v71/v72 weights.'
    correction=state.get('upload_template_correction',{})
    correction.update(official_score_user_reported=49.8,
        platform_feedback='2026-10-21: user reports v68 score 49.8; format passed.',
        official_score_availability='Reported by user 2026-10-21')
    state['releases']['56']=dict(csv=str(baseline.relative_to(ROOT)).replace('\\','/'),sha256=sha(baseline),
        official_score_user_reported=50.603,status='BEST_USER_REPORTED_BASELINE')
    state['releases']['68'].update(official_score_user_reported=49.8,status='LOWER_THAN_V56_USER_REPORTED')
    state['releases']['72']=dict(csv=release['file'],sha256=sha(candidate),status='READY_AWAITING_PLATFORM',
        baseline='v56',official_score=None,changed_tasks=changed,unchanged_tasks=20,
        independent_model_reload='PASS',local_archive=archive_meta,
        desktop_upload_copy=str(delivery),baseline_contains_historical_pretraining=True)
    save(ROOT/'CURRENT_STATUS.json',state)
    result=dict(status='PASS',candidate=str(candidate),upload_copy=str(delivery),rollback_copy=str(rollback),
        sha256=sha(candidate),archive=str(archive),archive_sha256=sha(archive),archive_members_verified=True,
        changed_tasks=changed,unchanged_tasks=20,official_score=None)
    save(ROOT/'docs/v72_delivery.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
