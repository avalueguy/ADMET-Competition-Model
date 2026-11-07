"""Verify downloaded v94 release and create a byte-identical upload copy."""
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
    out=ROOT/'admet_model_outputs_v94';conf=load(out/'confirmation.json');audit=load(out/'model_verification.json');accepted=conf['accepted'];assert audit['status']=='PASS'
    prefix=('v94_isolated_assay_' if accepted else 'v94_rejected_isolated_assay_')+'20261025';archive=load(ROOT/'releases'/(prefix+'.archive.json'));local=load(ROOT/'docs/v94_local_archive_verification.json');assert local['all_member_hashes_verified'] and local['archive_sha256']==archive['sha256']
    report=dict(version='v94',baseline='v92',baseline_official_score_user_reported=50.915,accepted=accepted,model_replay='PASS',checkpoints=audit['checkpoints'],archive=archive,all_archive_member_hashes_verified=True,official_score=None,automatically_uploaded=False,server_shutdown=False,new_csv=bool(accepted),invalid_attempt1_discarded=True)
    if accepted:
        rel=load(out/'release.json');assert rel['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM' and sha(out/'model_verification.json')==rel['verification_sha256'];file=ROOT/rel['file'];assert sha(file)==rel['sha256'];anchor=ROOT/'admet_submissions/v92_full_kernel/我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv';assert sha(anchor)==rel['anchor_sha256'];old,actual=rows(anchor),rows(file);changed=Counter()
        for a,z in zip(old[1:],actual[1:]):
            assert a[:4]+a[5:]==z[:4]+z[5:]
            if a[4]!=z[4]:changed[z[0]]+=1
            if z[0] not in accepted:assert a==z
        assert dict(changed)==rel['changed_tasks'];template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template);raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);chunks=[];cursor=0
        for i in range(1,len(actual)):
            lo,hi=ss[i][4];a,z=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
        chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
        dest=Path('D:/bili_download/v94_isolated_assay');dest.mkdir(parents=True,exist_ok=True);upload=dest/file.name
        if upload.exists():assert upload.read_bytes()==raw
        else:shutil.copy2(file,upload)
        assert sha(upload)==rel['sha256'];shutil.copy2(file.with_suffix('.verification.json'),upload.with_suffix('.verification.json'));shutil.copy2(ROOT/'docs/ITERATION_V94_20261025.md',dest/'v94_说明.md')
        report.update(status='LOCAL_DELIVERY_COMPLETE_AWAITING_PLATFORM',csv=rel['file'],upload_copy=str(upload),sha256=sha(upload),changed_tasks=dict(changed),unchanged_tasks=22-len(changed),rows=fmt['rows'],columns=6,tasks=22,nonprediction_bytes_exact=True);save(dest/'v94_delivery.json',report)
    else:report.update(status='REJECTED_NO_NEW_CSV',reason='No task passes fixed development and confirmation requirements.')
    save(ROOT/'docs/v94_local_delivery.json',report);state=load(ROOT/'CURRENT_STATUS.json');state.update(active_experiment='v94',active_experiment_status=report['status'],running_new_training_processes=0,running_new_iteration_processes_at_completion=0,new_submission_created=bool(accepted),best_official_version='v92',best_official_version_user_reported='v92',best_official_score_user_reported=50.915,official_score=50.915)
    state['v94_result']=report;state['active_iteration']=dict(version='v94',baseline='v92',highest_official_version='v92',highest_official_score=50.915,status=report['status'],csv=report.get('csv'),running_training_processes=0,official_score=None,github_push=False,server_shutdown_requested=False)
    if accepted:
        state.update(latest_completed_model_branch='v94',latest_completed_model_branch_is_proven_best=False,recommended_submission_version='v94',priority_next_step='User evaluate verified v94; do not infer official points from local results.')
        state['releases']['94']=dict(csv=report['csv'],sha256=report['sha256'],status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',official_score=None,baseline='v92',changed_tasks=report['changed_tasks'],unchanged_tasks=report['unchanged_tasks'],model_verification='PASS',archive=archive,desktop_upload_copy=report['upload_copy'])
    else:state['priority_next_step']='No new eligible submission; v92 remains official best. Broader model/validation redesign needed.'
    save(ROOT/'CURRENT_STATUS.json',state)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8-sig');marker='以下为历史记录；当前状态以本段与CURRENT_STATUS.json为准。';tail=text.split(marker,1)[1]
    top='# 当前官方最佳 v92（50.915）；v94 '+('待平台验证' if accepted else '未产生合格候选')+'\n\n用户最新报告v92官方50.915，比v88提高0.005。本轮完成更充分训练的共享结构网络v93，以及独立目标任务条件神经网络v94。v93两项候选确认失败，未生成CSV。v94第一次共享测量尝试发现跨任务输入隔离问题，已作废并保留记录；最终实验各目标网络独立，全部输入来源排除目标任务。\n\n'
    if accepted:top+=f"v94通过确认的任务：{', '.join(accepted)}，更新{sum(report['changed_tasks'].values())}条预测，其余任务保留v92预测字符串。CPU重载、源测量重建、目标查询标签扰动不变性、融合复算与模板非预测字节全部通过。\n\n"
    else:top+='v94未通过固定确认门槛，不生成改名或未经验证的提交文件。\n\n'
    top+='历史代理不是完整当前集成OOF，标签已多轮复用。本地指标无法换算官方涨幅，不能承诺达到52。\n\n[v93实验](docs/ITERATION_V93_20261025.md) · [v94实验](docs/ITERATION_V94_20261025.md) · [本地交付报告](docs/v94_local_delivery.json)\n\n';readme.write_text(top+marker+tail,encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
