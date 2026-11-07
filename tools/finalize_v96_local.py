"""Verify the downloaded v96 archive and create an exact local upload copy."""
from pathlib import Path
from collections import Counter
import csv,hashlib,io,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from verify_submission import verify
from repack_upload_template import field_spans

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))

def main():
    out=ROOT/'admet_model_outputs_v96';conf=load(out/'confirmation.json');audit=load(out/'model_verification.json')
    accepted=conf['accepted'];assert audit['status']=='PASS'
    prefix=('v96_normalized_assays_' if accepted else 'v96_rejected_normalized_assays_')+'20261026'
    archive=load(ROOT/'releases'/(prefix+'.archive.json'));local=load(ROOT/'docs/v96_local_archive_verification.json')
    assert local['all_member_hashes_verified'] and local['archive_sha256']==archive['sha256']
    report=dict(version='v96',baseline='v94',baseline_official_score_user_reported=50.916,accepted=accepted,model_replay='PASS',
        checkpoints=audit['checkpoints'],archive=archive,all_archive_member_hashes_verified=True,official_score=None,
        automatically_uploaded=False,server_shutdown=False,new_csv=bool(accepted))
    if accepted:
        rel=load(out/'release.json');assert rel['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM'
        assert sha(out/'model_verification.json')==rel['verification_sha256'];file=ROOT/rel['file'];assert sha(file)==rel['sha256']
        anchor=next((ROOT/'admet_submissions/v94_isolated_assay').glob('*.csv'));assert sha(anchor)==rel['anchor_sha256']
        old,actual=rows(anchor),rows(file);changed=Counter()
        for a,z in zip(old[1:],actual[1:]):
            assert a[:4]+a[5:]==z[:4]+z[5:]
            if a[4]!=z[4]:changed[z[0]]+=1
            if z[0] not in accepted:assert a==z
        assert dict(changed)==rel['changed_tasks'];template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template)
        raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);chunks=[];cursor=0
        for i in range(1,len(actual)):
            lo,hi=ss[i][4];a,z=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
        chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
        dest=Path('D:/bili_download/v96_normalized_assays');dest.mkdir(parents=True,exist_ok=True);upload=dest/file.name
        if upload.exists():assert upload.read_bytes()==raw
        else:shutil.copy2(file,upload)
        assert sha(upload)==rel['sha256'];shutil.copy2(file.with_suffix('.verification.json'),upload.with_suffix('.verification.json'))
        shutil.copy2(ROOT/'docs/ITERATION_V96_20261026.md',dest/'v96_说明.md')
        report.update(status='LOCAL_DELIVERY_COMPLETE_AWAITING_PLATFORM',csv=rel['file'],upload_copy=str(upload),sha256=sha(upload),
            changed_tasks=dict(changed),unchanged_tasks=22-len(changed),rows=fmt['rows'],columns=6,tasks=22,nonprediction_bytes_exact=True)
        save(dest/'v96_delivery.json',report)
    else:report.update(status='REJECTED_NO_NEW_CSV',reason='No endpoint passes frozen development and confirmation requirements.')
    save(ROOT/'docs/v96_local_delivery.json',report);state=load(ROOT/'CURRENT_STATUS.json')
    state.update(active_experiment='v96',active_experiment_status=report['status'],running_new_training_processes=0,
        running_new_iteration_processes_at_completion=0,new_submission_created=bool(accepted),best_official_version='v94',
        best_official_version_user_reported='v94',best_official_score_user_reported=50.916,official_score=50.916)
    state['v96_result']=report;state['active_iteration']=dict(version='v96',baseline='v94',highest_official_version='v94',highest_official_score=50.916,
        status=report['status'],csv=report.get('csv'),running_training_processes=0,official_score=None,github_push=False,server_shutdown_requested=False)
    if accepted:
        state.update(latest_completed_model_branch='v96',latest_completed_model_branch_is_proven_best=False,recommended_submission_version='v96',
            priority_next_step='Evaluate independently verified v96 on platform; local changes do not establish official gain.')
        state['releases']['96']=dict(csv=report['csv'],sha256=report['sha256'],status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',official_score=None,
            baseline='v94',changed_tasks=report['changed_tasks'],unchanged_tasks=report['unchanged_tasks'],model_verification='PASS',archive=archive,desktop_upload_copy=report['upload_copy'])
    else:state['priority_next_step']='v96 rejected; v94 remains official best. Broader model or feature redesign required.'
    save(ROOT/'CURRENT_STATUS.json',state)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8-sig');marker='以下为历史记录；当前状态以本段与CURRENT_STATUS.json为准。'
    tail=text.split(marker,1)[1]
    top='# 当前官方最佳 v94（50.916）；v96 '+('待平台验证' if accepted else '未产生合格候选')+'\n\n'
    top+='用户报告v94官方50.916，仅增加0.001。本轮已完成22任务中和母体测量迁移训练，两个种子、固定检查点、fold0/1开发冻结及fold2/valid确认。\n\n'
    if accepted:top+=f"v96通过确认：{', '.join(accepted)}，更新{sum(report['changed_tasks'].values())}条预测。CPU重载、测量来源重建、查询标签扰动不变性、融合及模板字节检查通过。\n\n"
    else:top+='没有任务满足固定确认要求，未生成改名的旧提交或不合格候选。\n\n'
    top+='目标52仍未达到。历史代理不是完整当前集成OOF，标签已反复复用，本地增益不能换算成官方分数。\n\n[v96实验](docs/ITERATION_V96_20261026.md) · [本地交付](docs/v96_local_delivery.json)\n\n'
    readme.write_text(top+marker+tail,encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
