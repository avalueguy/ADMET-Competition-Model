"""Verify restored release and exact template bytes, copy upload, record status."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,io,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from verify_submission import verify
from repack_upload_template import field_spans

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))

def main(version):
    out=ROOT/f'admet_model_outputs_v{version}';rel=read(out/'release.json');audit=read(out/'model_verification.json')
    prefix=f'v{version}_multi_route_20261026';archive=read(ROOT/'releases'/(prefix+'.archive.json'))
    local=read(ROOT/f'docs/v{version}_local_archive_verification.json')
    assert local['all_member_hashes_verified'] and local['dependency_hashes_verified'] and local['archive_sha256']==archive['sha256']
    assert audit['status']=='PASS' and rel['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM'
    file=ROOT/rel['file'];assert sha(file)==rel['sha256']
    anchor=next((ROOT/'admet_submissions/v94_isolated_assay').glob('*.csv'));assert sha(anchor)==rel['anchor_sha256']
    old,actual=rows(anchor),rows(file);changed=Counter();assert len(old)==len(actual)
    for a,z in zip(old[1:],actual[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[4]!=z[4]:changed[z[0]]+=1
        if z[0] not in rel['selected']:assert a==z
    assert dict(changed)==rel['changed_tasks']
    template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template)
    raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);chunks=[];cursor=0
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];a,z=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    dest=Path(f'D:/bili_download/v{version}_multi_route');dest.mkdir(parents=True,exist_ok=True);upload=dest/file.name
    if upload.exists():assert upload.read_bytes()==raw
    else:shutil.copy2(file,upload)
    assert sha(upload)==rel['sha256'];shutil.copy2(file.with_suffix('.verification.json'),upload.with_suffix('.verification.json'))
    shutil.copy2(ROOT/f'docs/ITERATION_V{version}_20261026.md',dest/f'v{version}_说明.md')
    report=dict(status='LOCAL_DELIVERY_COMPLETE_AWAITING_PLATFORM',version=f'v{version}',baseline='v94',baseline_official_score_user_reported=50.916,
        csv=rel['file'],upload_copy=str(upload),sha256=sha(upload),changed_tasks=dict(changed),changed_predictions=sum(changed.values()),
        unchanged_tasks=22-len(changed),rows=fmt['rows'],columns=6,tasks=22,nonprediction_bytes_exact=True,model_replay='PASS',
        archive=archive,all_archive_member_hashes_verified=True,dependencies_verified=True,official_score=None,target_score=52,target_achieved=False,
        automatically_uploaded=False,server_shutdown=False,github_push=False)
    save(dest/f'v{version}_delivery.json',report);save(ROOT/f'docs/v{version}_local_delivery.json',report)
    state=read(ROOT/'CURRENT_STATUS.json');state.update(handoff_date='2026-10-26',active_experiment=f'v{version}',active_experiment_status=report['status'],
        running_new_training_processes=0,running_new_iteration_processes_at_completion=0,new_submission_created=True,
        best_official_version='v94',best_official_version_user_reported='v94',best_official_score_user_reported=50.916,official_score=50.916,
        latest_completed_model_branch=f'v{version}',latest_completed_model_branch_is_proven_best=False,recommended_submission_version=f'v{version}',
        target_achieved=False,priority_next_step=f'Evaluate v{version} on platform; local gains do not establish official gain.')
    state['active_iteration']=dict(version=f'v{version}',baseline='v94',highest_official_version='v94',highest_official_score=50.916,
        status=report['status'],csv=report['csv'],running_training_processes=0,official_score=None,github_push=False,server_shutdown_requested=False)
    state[f'v{version}_result']=report;state['releases'][str(version)]=dict(csv=report['csv'],sha256=report['sha256'],status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',
        official_score=None,baseline='v94',changed_tasks=dict(changed),unchanged_tasks=22-len(changed),model_verification='PASS',archive=archive,desktop_upload_copy=str(upload))
    for n in (95,96,98,100):
        conf=read(ROOT/f'admet_model_outputs_v{n}/confirmation.json');state[f'v{n}_experiment_result']=dict(status='COMPLETE',accepted=conf['accepted'],official_score=None)
    save(ROOT/'CURRENT_STATUS.json',state)
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8-sig');marker='以下为历史记录；当前状态以本段与CURRENT_STATUS.json为准。';tail=text.split(marker,1)[1]
    top=f'# 当前官方最佳 v94（50.916）；v{version}待平台验证\n\n'
    top+='本轮完成独立任务图网络、化学身份标准化测量迁移、三维构象树与三维结构核四条路线；开发fold0/1冻结，fold2/valid确认，无失败后换候补。\n\n'
    top+=f"最终更新{len(changed)}项任务、{sum(changed.values())}条预测：{', '.join(changed)}。模型重载、输入重建、目标查询标签扰动、融合复算和模板非预测字节检查通过。归档全部成员及依赖哈希本地验证通过。\n\n"
    top+='目标52仍未达到。历史代理不是精确当前集成OOF，标签已多次复用，本地收益不能换算官方涨幅。没有自动提交、push GitHub或关闭服务器。\n\n'
    top+=f'[本轮实验](docs/ITERATION_V{version}_20261026.md) · [本地交付](docs/v{version}_local_delivery.json)\n\n'
    p.write_text(top+marker+tail,encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('version',type=int,choices=[97,99,101]);args=ap.parse_args();main(args.version)
