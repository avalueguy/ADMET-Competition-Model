"""Verify downloaded archive, extract exact artifacts, and deliver v80 locally."""
from pathlib import Path,PurePosixPath
import sys
import csv
import io
import json
import tarfile
import hashlib
import shutil
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from verify_submission import verify
from repack_upload_template import field_spans
PREFIX='v80_cyp2c9_residual_20261023'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def rows(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))

def main():
    meta=load(ROOT/'releases'/(PREFIX+'.archive.json'));archive=ROOT/meta['path']
    assert sha(archive)==meta['sha256'] and archive.stat().st_size==meta['bytes']
    manifest_path=ROOT/'releases'/(PREFIX+'.manifest.json');manifest=load(manifest_path)
    expected={r['path']:r for r in manifest['files']};seen=set()
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            rel=PurePosixPath(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and member.name not in seen
            seen.add(member.name);raw=tf.extractfile(member).read()
            if member.name==manifest_path.relative_to(ROOT).as_posix():
                assert raw==manifest_path.read_bytes();continue
            rec=expected[member.name]
            assert len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['sha256']
            dest=(ROOT/member.name).resolve();assert dest.is_relative_to(ROOT.resolve())
            if dest.exists():assert sha(dest)==rec['sha256'],member.name
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    assert seen==set(expected)|{manifest_path.relative_to(ROOT).as_posix()} and len(seen)==meta['members']
    out=ROOT/'admet_model_outputs_v80_final';release=load(out/'release.json')
    audit=load(out/'model_verification.json')
    assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM' and audit['status']=='PASS' and audit['models']==29
    assert sha(out/'model_verification.json')==release['verification_sha256']
    for rec in audit['model_records']:assert sha(ROOT/rec['path'])==rec['sha256']
    for path,expected_hash in release['input_hashes'].items():assert sha(ROOT/path)==expected_hash,path
    file=ROOT/release['file'];template=ROOT/'admet_official/submission_template.csv'
    assert sha(file)==release['sha256']
    actual=rows(file);basefile=next((ROOT/'admet_submissions/v73_v72').glob('*.csv'));base=rows(basefile)
    assert sha(basefile)==release['anchor_sha256']
    changed=Counter()
    for row,old in zip(actual[1:],base[1:]):
        assert row[:4]+row[5:]==old[:4]+old[5:]
        if row[4]!=old[4]:changed[row[0]]+=1
        if row[0]!='cyp2c9_veith':assert row==old
    assert dict(changed)=={'cyp2c9_veith':2419}
    fmt=verify(file,template)
    raw=file.read_bytes();ref=template.read_bytes();s=field_spans(raw);t=field_spans(ref);chunks=[];cursor=0
    for i in range(1,len(actual)):
        lo,hi=s[i][4];a,b=t[i][4];chunks.extend((raw[cursor:lo],ref[a:b]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    assert fmt['rows']==16379 and fmt['tasks']==22 and fmt['utf8_bom'] and fmt['crlf_only']
    dest=Path('D:/bili_download/v80_cyp2c9_residual');dest.mkdir(parents=True,exist_ok=True)
    upload=dest/file.name
    if upload.exists():assert upload.read_bytes()==raw
    else:shutil.copy2(file,upload)
    shutil.copy2(file.with_suffix('.verification.json'),upload.with_suffix('.verification.json'))
    shutil.copy2(ROOT/'docs/ITERATION_V80_20261023.md',dest/'v80_说明.md')
    assert sha(upload)==release['sha256']
    delivery=dict(status='LOCAL_DELIVERY_COMPLETE_AWAITING_PLATFORM',version='v80',csv=release['file'],
        upload_copy=str(upload),sha256=sha(file),changed_tasks=dict(changed),unchanged_tasks=21,
        rows=16379,columns=6,tasks=22,model_reload='PASS',models=29,final_models=3,
        nonprediction_bytes_exact=True,archive=meta,all_archive_member_hashes_verified=True,
        official_score=None,best_official_score_user_reported=50.640,best_official_version_user_reported='v73',
        test_labels_used=False,external_data_new_branch=False,pretrained_weights_new_branch=False,
        anchor_contains_historical_pretraining=True,automatic_upload=False,github_push=False,
        nested_results=audit['nested'],meta_diagnostics=audit['meta_diagnostics'])
    save(ROOT/'docs/v80_delivery.json',delivery);save(dest/'v80_delivery.json',delivery)
    state=load(ROOT/'CURRENT_STATUS.json')
    state.update(handoff_date='2026-10-23',latest_completed_model_branch='v80',
        latest_completed_model_branch_is_proven_best=False,recommended_submission_version='v80',
        best_official_version='v73',best_official_version_user_reported='v73',best_official_score_user_reported=50.640,
        active_experiment='v80',active_experiment_status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',
        new_submission_created=True,running_new_training_processes=0,official_score=None,
        priority_next_step='Submit v80 and record platform score for its exact file hash.')
    state['active_iteration']=dict(version='v80',status=delivery['status'],csv=release['file'],
        changed_tasks=dict(changed),independent_model_verification='PASS',archive_verified=True,
        running_training_processes=0,official_score=None,local_delivery_report='docs/v80_delivery.json',
        local_gpu_training=False,github_push=False,server_shutdown_requested=False)
    state['releases']['80']=dict(csv=release['file'],sha256=sha(file),
        status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',official_score=None,baseline='v73',
        changed_tasks=dict(changed),unchanged_tasks=21,model_verification='PASS',
        archive=meta,desktop_upload_copy=str(upload))
    state['v80_result']=dict(accepted=True,official_score=None,new_csv=True,models_verified=29,
        nested=audit['nested'],meta_diagnostics=audit['meta_diagnostics'])
    save(ROOT/'CURRENT_STATUS.json',state)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    marker='<!-- current-feedback-v76 -->';history=text.find('# ADMET Competition Model — 2026-10-22')
    if text.startswith(marker) and history>=0:text=text[history:]
    if not text.startswith('<!-- current-candidate-v80 -->'):
        intro=('<!-- current-candidate-v80 -->\n# 最新提交候选：v80（等待官方评分）\n\n'
            'v80 已完成：只替换 CYP2C9 的 2,419 条预测，其他 21 任务保持最好已知结果。'
            '三折及官方 valid 代理检查均正增益；两项嵌套对照 AUPRC 分别增加 0.012842、0.013594。'
            '29 个模型独立重载、CSV 非预测字节与本地归档校验通过。'
            '当前最高官方成绩仍是用户报告的 v73：50.640，不能承诺 v80 官方提分。\n\n'
            '[v80 提交文件](admet_submissions/v80_cyp2c9_residual/) · '
            '[实验与限制](docs/ITERATION_V80_20261023.md) · [交付校验](docs/v80_delivery.json)\n\n'
            '以下为历史记录；当前状态以本段与 CURRENT_STATUS.json 为准。\n\n')
        readme.write_text(intro+text,encoding='utf-8')
    print(json.dumps(delivery,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
