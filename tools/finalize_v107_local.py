"""Verify restored evidence and copy exactly one candidate to upload directory."""
from pathlib import Path
from collections import Counter
import csv,hashlib,io,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from verify_submission import verify
from repack_upload_template import field_spans
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def main():
    rel=read(ROOT/'admet_model_outputs_v107/release.json');audit=read(ROOT/'admet_model_outputs_v107/model_verification.json')
    archive=read(ROOT/'releases/v107_count_difference_boost_20261026.archive.json');local=read(ROOT/'docs/v107_local_archive_verification.json')
    assert rel['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM' and audit['status']=='PASS'
    assert local['all_member_hashes_verified'] and local['dependency_hashes_verified'] and local['archive_sha256']==archive['sha256']
    file=ROOT/rel['file'];assert sha(file)==rel['sha256'];anchor=next((ROOT/'admet_submissions/v101_graph_assay_shape').glob('*.csv'));assert sha(anchor)==rel['anchor_sha256']
    old,actual=rows(anchor),rows(file);changed=Counter();assert len(old)==len(actual)
    for a,z in zip(old[1:],actual[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[4]!=z[4]:changed[z[0]]+=1
        if z[0] not in rel['selected']:assert a==z
    assert dict(changed)==rel['changed_tasks'];template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template)
    raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);chunks=[];cursor=0
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];a,z=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    dest=Path('D:/bili_download/v107_count_difference_boost');dest.mkdir(parents=True,exist_ok=True);upload=dest/file.name
    if upload.exists():assert upload.read_bytes()==raw
    else:shutil.copy2(file,upload)
    assert sha(upload)==rel['sha256'];shutil.copy2(file.with_suffix('.verification.json'),upload.with_suffix('.verification.json'))
    shutil.copy2(ROOT/'docs/ITERATION_V107_20261026.md',dest/'v107_说明.md')
    report=dict(status='LOCAL_DELIVERY_COMPLETE_AWAITING_PLATFORM',version='v107',baseline='v101',baseline_official_score_user_reported=50.932,
        csv=rel['file'],upload_copy=str(upload),sha256=rel['sha256'],changed_tasks=dict(changed),changed_predictions=sum(changed.values()),unchanged_tasks=22-len(changed),
        rows=fmt['rows'],columns=6,tasks=22,nonprediction_bytes_exact=True,model_replay='PASS',archive=archive,all_archive_member_hashes_verified=True,dependencies_verified=True,
        official_score=None,target_score=52,target_achieved=False,automatically_uploaded=False,github_push=False,server_shutdown=False)
    save(dest/'v107_delivery.json',report);save(ROOT/'docs/v107_local_delivery.json',report)
    p=ROOT/'CURRENT_STATUS.json';state=read(p);state.update(handoff_date='2026-10-26',active_experiment='v107',active_experiment_status=report['status'],running_new_training_processes=0,
        running_new_iteration_processes_at_completion=0,new_submission_created=True,best_official_version='v101',best_official_version_user_reported='v101',best_official_score_user_reported=50.932,official_score=50.932,
        latest_completed_model_branch='v107',latest_completed_model_branch_is_proven_best=False,recommended_submission_version='v107',target_achieved=False,priority_next_step='Evaluate v107 on platform. No official-score guarantee.')
    state['active_iteration']=dict(version='v107',baseline='v101',highest_official_version='v101',highest_official_score=50.932,status=report['status'],csv=report['csv'],running_training_processes=0,official_score=None,github_push=False,server_shutdown_requested=False)
    state['v107_result']=report;state['releases']['107']=dict(csv=report['csv'],sha256=report['sha256'],status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',official_score=None,baseline='v101',changed_tasks=dict(changed),archive=archive,desktop_upload_copy=str(upload),model_verification='PASS')
    for n in (102,103,104,106):state[f'v{n}_experiment_result']=dict(status='COMPLETE',accepted=read(ROOT/f'admet_model_outputs_v{n}/confirmation.json')['accepted'],official_score=None)
    save(p,state)
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8-sig');marker='以下为历史记录；当前状态以本段与CURRENT_STATUS.json为准。';tail=text.split(marker,1)[1]
    top='# 当前官方最佳 v101（50.932）；v107待平台验证\n\n本轮测试计数SVM/SVR、参照分子性质差值、未折叠子结构margin与稀疏提升四条路线。开发fold0/1冻结后确认，失败不换候补。\n\n'
    top+=f"最终更新{len(changed)}项、{sum(changed.values())}条预测：{', '.join(changed)}。模型独立重载、输入重建、查询标签扰动与模板非预测字节检查通过，本地归档全部成员及依赖哈希验证通过。\n\n"
    top+='尚无v107官方成绩，不能承诺达到52。本地验证沿用近似历史代理且标签已多次复用。没有自动提交、pushGitHub或关闭服务器。\n\n[实验记录](docs/ITERATION_V107_20261026.md) · [本地交付](docs/v107_local_delivery.json)\n\n'
    p.write_text(top+marker+tail,encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
