"""Local archive, predictions and template verification; no model training."""
from pathlib import Path,PurePosixPath
import sys,json,hashlib,tarfile,shutil
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from verify_submission import verify,read
from repack_upload_template import field_spans
OUT=ROOT/'admet_model_outputs_v75';PREFIX='v75_v73_20261022'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def blend(metric,a,b,w):
    if metric!='MAE':a,b=rankdata(a)/len(a),rankdata(b)/len(b)
    return (1-w)*a+w*b


def main():
    meta=load(ROOT/'releases'/(PREFIX+'.archive.json'));archive=ROOT/meta['path'];manifest_path=ROOT/'releases'/(PREFIX+'.manifest.json')
    assert sha(archive)==meta['sha256'] and archive.stat().st_size==meta['bytes']
    manifest=load(manifest_path);expected={r['path']:r for r in manifest['files']};seen=set();manifest_name=manifest_path.relative_to(ROOT).as_posix()
    with tarfile.open(archive,'r|gz') as tf:
        for m in tf:
            rel=PurePosixPath(m.name);assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts and m.name not in seen
            seen.add(m.name);raw=tf.extractfile(m).read()
            if m.name==manifest_name:assert raw==manifest_path.read_bytes();continue
            r=expected[m.name];assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
            p=(ROOT/m.name).resolve();assert p.is_relative_to(ROOT.resolve())
            if m.name=='admet_model_outputs_v75/pipeline_status.json':continue
            if m.name.startswith(('admet_model_outputs_v75/','admet_submissions/v75_v73/')):
                if p.exists():assert p.read_bytes()==raw, m.name
                else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            elif m.name.endswith('.py') or m.name=='admet_v75_protocol.json':assert p.is_file() and sha(p)==r['sha256'],m.name
    assert seen==set(expected)|{manifest_name} and len(seen)==meta['members']
    print(json.dumps(dict(stage='ARCHIVE_VERIFIED',members=len(seen))),flush=True)
    release=load(OUT/'release.json');audit=load(OUT/'model_verification.json');confirmed=load(OUT/'confirmation.json');protocol=load(ROOT/'admet_v75_protocol.json')
    assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM' and audit['status']=='PASS'
    assert sha(OUT/'model_verification.json')==release['verification_sha256']
    assert release['signature']==audit['signature']==confirmed['signature']
    for r in audit['checks']:assert sha(ROOT/r['path'])==r['sha256']
    file=ROOT/release['file'];base=next((ROOT/'admet_submissions/v73_v72').glob('*.csv'));template=ROOT/'admet_official/submission_template.csv'
    assert sha(file)==release['sha256'];assert sha(base)==release['anchor_sha256']==protocol['anchor_sha256']
    assert sha(template)=='8ee41589d2c98e7b93081c058b3f928ac09bea4b3ffc899fa13677051e9baa20'
    fmt=verify(file,template);rows,old=read(file),read(base);replay=load(OUT/'fusion_replay.json');changed={}
    for task in dict.fromkeys(r[0] for r in rows[1:]):
        actual=[r for r in rows[1:] if r[0]==task];prior=[r for r in old[1:] if r[0]==task]
        if task not in release['selected']:assert actual==prior;continue
        selected=release['selected'][task];choice=selected['choice'];metric=selected['metric'];conditional=None
        if choice['conditional']:
            conditional=np.mean([np.load(OUT/task/'final'/f'conditional_{seed}/predictions.npz')['p'] for seed in protocol['final_seeds']],axis=0)
        expert=conditional
        if choice['kernel'] is not None:
            raw=np.load(OUT/task/'final/kernels/predictions.npz')['p'][choice['kernel']]
            expert=raw if conditional is None else blend(metric,raw,conditional,choice['conditional'])
        anchor=np.array([float(r[4]) for r in prior]);expected_values=blend(metric,anchor,expert,protocol['anchor_weight'])
        if task=='ppbr_az':expected_values=expected_values.clip(0,100)
        np.testing.assert_array_equal(expected_values,np.array([float(r[4]) for r in actual]))
        np.testing.assert_array_equal(anchor,replay[task]['anchor']);np.testing.assert_array_equal(expert,replay[task]['expert']);np.testing.assert_array_equal(expected_values,replay[task]['result'])
        changed[task]=sum(r[4]!=s[4] for r,s in zip(actual,prior))
    assert changed==audit['changed_tasks'] and any(changed.values())
    raw,ref=file.read_bytes(),template.read_bytes();s,t=field_spans(raw),field_spans(ref);chunks=[];cursor=0
    for i in range(1,len(rows)):
        lo,hi=s[i][4];a,z=t[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    assert fmt['rows']==16379 and fmt['tasks']==22 and fmt['utf8_bom'] and fmt['crlf_only']
    dest=Path('D:/bili_download/v75_v73');dest.mkdir(parents=True,exist_ok=True);upload=dest/file.name
    if upload.exists():assert upload.read_bytes()==file.read_bytes()
    else:shutil.copy2(file,upload)
    fmt.update(independent_model_verification='PASS',independent_local_fusion_replay='PASS',nonprediction_bytes_exact=True,changed_tasks=changed)
    save(upload.with_suffix('.verification.json'),fmt)
    result=dict(status='PASS_LOCAL_DELIVERY_AWAITING_PLATFORM',candidate=release['file'],upload_copy=str(upload),sha256=sha(file),
        rows=16379,columns=6,tasks=22,format_verification='PASS',template_sha256=sha(template),independent_model_verification='PASS',
        models=audit['models'],final_models=audit['final_models'],independent_local_fusion_replay='PASS',nonprediction_bytes_exact=True,
        archive=meta,all_archive_member_hashes_verified=True,changed_tasks=changed,unchanged_tasks=22-len(changed),official_score=None,
        anchor='v73',anchor_official_score_user_reported=50.640,exact_full_anchor_oof=False,conditional_assay_evaluation=True,
        test_labels_used=False,external_data_new_branch=False,pretrained_new_branch=False,anchor_contains_historical_pretraining=True,automatic_upload=False)
    save(ROOT/'docs/v75_delivery.json',result)
    state=load(ROOT/'CURRENT_STATUS.json');state.update(latest_completed_model_branch='v75',latest_completed_model_branch_is_proven_best=False,
        rollback_baseline='v73',official_score=None,best_official_version_user_reported='v73',best_official_score_user_reported=50.640,
        priority_next_step='Submit v75 manually and record platform score; v73 remains best reported baseline.',running_new_iteration_processes_at_completion=0)
    state['active_iteration'].update(status='LOCAL_DELIVERY_COMPLETE_AWAITING_PLATFORM',changed_tasks=changed,independent_model_verification='PASS',archive_verified=True,running_training_processes=0)
    state['releases']['75']=dict(csv=release['file'],sha256=sha(file),status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',baseline='v73',official_score=None,
        changed_tasks=changed,unchanged_tasks=22-len(changed),model_verification='PASS',archive=meta,desktop_upload_copy=str(upload))
    save(ROOT/'CURRENT_STATUS.json',state);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
