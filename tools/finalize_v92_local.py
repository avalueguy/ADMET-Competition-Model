"""Standard-library local release verification and exact upload copy."""
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
    out=ROOT/'admet_model_outputs_v92';rel=load(out/'release.json');audit=load(out/'model_verification.json');archive=load(ROOT/'releases/v92_full_kernel_20261025.archive.json');local=load(ROOT/'docs/v92_local_archive_verification.json')
    assert rel['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM' and audit['status']=='PASS' and local['all_member_hashes_verified']
    assert sha(out/'model_verification.json')==rel['verification_sha256']
    file=ROOT/rel['file'];anchor=ROOT/'admet_submissions/v88_context_kernel/我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv';template=ROOT/'admet_official/submission_template.csv'
    assert sha(file)==rel['sha256'] and sha(anchor)==rel['anchor_sha256'];actual,old=rows(file),rows(anchor);changed=Counter()
    for a,z in zip(old[1:],actual[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[4]!=z[4]:changed[z[0]]+=1
        if z[0] not in rel['changed_tasks']:assert a==z
    assert dict(changed)==rel['changed_tasks'];fmt=verify(file,template);raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);cursor=0;chunks=[]
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];a,z=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    dest=Path('D:/bili_download/v92_full_kernel');dest.mkdir(parents=True,exist_ok=True);upload=dest/file.name
    if upload.exists():assert upload.read_bytes()==raw
    else:shutil.copy2(file,upload)
    shutil.copy2(file.with_suffix('.verification.json'),upload.with_suffix('.verification.json'));shutil.copy2(ROOT/'docs/ITERATION_V92_20261025.md',dest/'v92_说明.md')
    assert sha(upload)==rel['sha256']
    report=dict(status='LOCAL_DELIVERY_COMPLETE_AWAITING_PLATFORM',version='v92',csv=rel['file'],upload_copy=str(upload),sha256=sha(upload),
        changed_tasks=dict(changed),unchanged_tasks=20,rows=fmt['rows'],columns=6,tasks=22,nonprediction_bytes_exact=True,
        model_replay='PASS',kernel_models=10,all_archive_member_hashes_verified=True,archive=archive,official_score=None,
        baseline='v88',baseline_official_score_user_reported=50.910,bootstrap_intervals_cross_zero=True,test_labels_used=False,
        external_data_new_branch=False,new_pretrained_weights=False,automatically_uploaded=False,server_shutdown=False)
    save(ROOT/'docs/v92_local_delivery.json',report);save(dest/'v92_delivery.json',report)
    state=load(ROOT/'CURRENT_STATUS.json');state.update(latest_completed_model_branch='v92',latest_completed_model_branch_is_proven_best=False,
        recommended_submission_version='v92',best_official_version='v88',best_official_version_user_reported='v88',best_official_score_user_reported=50.910,
        active_experiment='v92',active_experiment_status=report['status'],new_submission_created=True,running_new_training_processes=0,
        running_new_iteration_processes_at_completion=0,official_score=None,priority_next_step='User submit verified v92 and report exact official score. Local gain remains uncertain; highest official v88 50.910. Do not retune on confirm labels.',server_shutdown_requested=False)
    state['active_iteration']=dict(version='v92',baseline='v88',highest_official_version='v88',highest_official_score=50.910,status=report['status'],
        candidate='v92',csv=rel['file'],changed_tasks=dict(changed),independent_model_replay='PASS',archive_verified=True,running_training_processes=0,
        official_score=None,local_delivery_report='docs/v92_local_delivery.json',github_push=False,server_shutdown_requested=False)
    state['v92_result'].update(status=report['status'],accepted=['ames','cyp2d6_veith'],new_csv=True,changed_predictions=4083,kernel_models_verified=10,
        bootstrap_95_intervals_cross_zero=True,official_score=None,archive=archive,all_member_hashes_verified=True)
    state['releases']['92']=dict(csv=rel['file'],sha256=rel['sha256'],status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',official_score=None,
        baseline='v88',changed_tasks=dict(changed),unchanged_tasks=20,model_verification='PASS',archive=archive,desktop_upload_copy=str(upload))
    state['releases']['88'].update(official_score=50.910,status='USER_REPORTED_OFFICIAL_BEST')
    save(ROOT/'CURRENT_STATUS.json',state)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8-sig');marker='以下为历史记录；当前状态以本段与CURRENT_STATUS.json为准。';tail=text.split(marker,1)[1]
    top='''<!-- current-candidate-v92 -->
# 当前候选 v92；最高已知官方版本 v88（50.910）

用户报告v88官方50.910，较v86提高0.010；平台仍只有总分。v90广覆盖分类残差树在确认失败，v91局部误差邻居22任务均未通过开发门槛，均不生成CSV。v92改用全量目标拟合分子的完整多指纹Tanimoto ridge，比较直接目标与历史残差，无新增外部数据或预训练权重。

22任务开发后冻结9项，确认只接受AMES与CYP2D6抑制。两项均为完整多视图直接核、alpha0.3、25%新专家；更新4083条预测，其余20任务保持v88。10份所用核模型CPU重算、拟合来源与目标变换、子批推理、最终融合与模板非预测字节全部通过，233项归档逐一哈希通过。

确认指标提高较小，bootstrap区间跨零；历史代理不是完整v88 OOF，标签历史复用。新候选没有官方分数，不能承诺大幅提分或达到52。服务器训练结束，未关闭服务器、未自动提交平台或推送GitHub。

[v92提交](admet_submissions/v92_full_kernel/) · [v92实验](docs/ITERATION_V92_20261025.md) · [本地交付校验](docs/v92_local_delivery.json) · [v90失败实验](docs/ITERATION_V90_20261025.md) · [v91失败实验](docs/ITERATION_V91_20261025.md) · [v88官方最好底稿](admet_submissions/v88_context_kernel/)

'''
    readme.write_text(top+marker+tail,encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
