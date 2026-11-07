"""Independent standard-library fusion verification and final delivery."""
from collections import Counter
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from finalize_v124_local import load,save,sha,rows,array,ranks
from verify_submission import verify
from repack_upload_template import field_spans
def main():
    report=load(ROOT/'admet_model_outputs_v130/evidence.json');audit=load(ROOT/'admet_model_outputs_v129/independent_verification.json')
    backup=load(ROOT/'releases/v130_low_task_experts_20261027.archive.json');local=load(ROOT/'docs/v130_local_archive_verification.json')
    assert audit['status']=='PASS' and local['all_member_hashes_verified'] and local['dependency_hashes_verified'] and backup['sha256']==local['archive_sha256']
    baseline=next((ROOT/'admet_submissions/v124_matched_models_history').glob('*.csv'));file=ROOT/report['file']
    assert sha(baseline)==report['baseline_sha256'] and sha(file)==report['sha256']
    original,actual=rows(baseline),rows(file);assert len(original)==len(actual)==16380
    for task,ch in report['selected'].items():
        number=ch['version'];r=ch['record']
        if number=='v128':
            cfg=r['config'];path=ROOT/f"admet_model_outputs_v128/models/{task}/final/{cfg['view']}_C{cfg['C']}.npz"
        else:path=ROOT/f"admet_model_outputs_{number}/models/{task}/final/{r['view']}/predictions.npz"
        expert=array(path,'p');pos=[i for i,row in enumerate(original) if i and row[0]==task]
        assert len(expert)==len(pos)
        old=[float(original[i][4]) for i in pos];weight=ch['weight']
        if original[pos[0]][5]=='MAE':p=[(1-weight)*a+weight*z for a,z in zip(old,expert)]
        else:p=[(1-weight)*a/len(old)+weight*z/len(old) for a,z in zip(ranks(old),ranks(expert))]
        assert all(abs(float(actual[i][4])-value)<1e-14 for i,value in zip(pos,p))
    changed=Counter()
    for a,z in zip(original[1:],actual[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[0] not in report['selected']:assert a==z
        if a[4]!=z[4]:changed[a[0]]+=1
    assert dict(changed)==report['verification']['changed_tasks']
    official=load(ROOT/'docs/platform_scorecard_v124_20261027.json');high=[r['task'] for r in official['tasks'] if r['score']>=51]
    assert not set(high)&set(changed)
    template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template)
    raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);pieces=[];cursor=0
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];a,z=tt[i][4];pieces.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    pieces.append(raw[cursor:]);assert b''.join(pieces)==ref
    assert fmt['rows']==16379 and fmt['tasks']==22 and fmt['utf8_bom'] and fmt['crlf_only']
    fmt.update(nonprediction_bytes_exact=True,independent_final_fusion_replay=True,protected_task_predictions_exact=True)
    # Keep packaged sidecars immutable; record the extra local replay separately.
    save(ROOT/'docs/v130_local_submission_verification.json',fmt)
    model_count=sum(len(r['model_hashes']) for r in audit['checks'])
    lines=['# v130：低分任务专项优化','',
        '当前最好官方版本是 v124：51.115。成绩单均值为 51.115090；v124 比 v114 提高 0.024072，其中半衰期贡献分项 +0.451124，AMES 贡献 +0.078452。v130 尚无官方成绩，52 分尚未达到。','',
        '本轮按用户要求，优先训练低分任务。官方分项达到 51 的任务全部保留 v124 原预测；其他未通过确认的任务也逐字保留。HIA 虽然归一化分项约 50.14，但 AUROC 已是 1，因此同样保留。','',
        '## 新模型与本地结果','',
        '| 任务 | 模型路线 | 分区 | 基线指标 | 新融合指标 | 增益 |',
        '|---|---|---|---:|---:|---:|']
    for r in report['validation_replay']:
        name=report['selected'][r['task']]['version'];lines.append(f"| {r['task']} | {name} | {r['part']} | {r['baseline']:.6f} | {r['candidate']:.6f} | {r['gain']:+.6f} |")
    lines+=['',
        'v126 尝试肝细胞清除和溶解度的双目标 CatBoost；v127 比较四个低分分类任务的 GPU 条件树；v128 比较相同低分分类任务的分子 Tanimoto 与排除目标任务的测量 RBF 核 SVM。开发 fold0/1 冻结方法，然后确认 fold2/valid。确认失败不更换候补或调整权重。多路线同时通过同一任务时，以开发折的最小增益选择，不以确认集成绩重新选型。','',
        'CYP3A4 底物的当前预测来自 v67 共享网络。已恢复并校验原归档，按原模型每个分区的目标训练行作匹配对照；其 valid 对照训练行只有 155，另一些历史代理实验使用 468 行，不能混为同一比较。其他任务虽然重建了当前分支最终预测，上游历史 OOF 仍有近似性；溶解度完整 v44 OOF 暂未恢复。','',
        '新模型仅使用官方 train/valid 标签、已有结构特征，以及排除了目标测量、目标 mask 的其他官方测量。test 结构来自模板，没有读取 test Y。开发标签历史上已反复使用；以下 bootstrap 只是不确定性诊断，不能消除选择偏差。','',
        '| 任务 | 路线 | 确认部分 | 骨架 bootstrap 增益 95% 区间 |','|---|---|---|---|']
    for r in audit['checks']:
        if 'scaffold_bootstrap_gain_ci95' in r:
            lo,hi=r['scaffold_bootstrap_gain_ci95'];lines.append(f"| {r['task']} | {r['version']} | {r['part']} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','## 通过与排除','']
    for r in audit['experiments']:lines.append(f"- {r['version']} 最终通过：{', '.join(r['accepted']) or '无'}。")
    lines+=['',f"最终改变 {len(changed)} 个任务、{sum(changed.values())} 条预测：{dict(changed)}。其他 {22-len(changed)} 个任务与 v124 原 CSV 预测字符串完全一致。",'',
        f'独立重载 {model_count} 个模型检查点通过；模型输入与查询标签扰动不变性、四分区指标、最终融合都已复算。CSV 共 16,379 行、22 任务、6 列，BOM/CRLF、drug_id 小数字符串和全部非预测字节与模板一致。', '',
        f"CSV SHA256：`{report['sha256']}`",'',
        f"增量备份：`{backup['path']}`，{backup['files']} 个成员全部哈希通过。依赖已哈希固定，包含 v124 历史包和 v125 基线数组包。历史基线预训练来源资格未在本轮重审。",'',
        '本地验证不能保证改变任务的官方成绩不下降；未改变任务的预测则已逐字保护。当前还缺约 0.885 总分，不能由这些本地增益推算官方达到 52。未自动提交、未 push GitHub、未关闭服务器。']
    doc=ROOT/'docs/ITERATION_V130_20261027.md';doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dest=Path('D:/bili_download/v130_low_task_experts');dest.mkdir(parents=True,exist_ok=True);upload=dest/file.name
    if upload.exists():assert upload.read_bytes()==raw
    else:shutil.copy2(file,upload)
    assert sha(upload)==report['sha256'];shutil.copy2(file.with_suffix('.verification.json'),upload.with_suffix('.verification.json'));shutil.copy2(doc,dest/'v130_说明.md')
    delivery=dict(status='LOCAL_DELIVERY_COMPLETE_AWAITING_PLATFORM',version='v130',baseline='v124',baseline_official_score=51.115,
        csv=report['file'],upload_copy=str(upload),sha256=report['sha256'],changed_tasks=dict(changed),changed_predictions=sum(changed.values()),
        unchanged_tasks=22-len(changed),high_score_tasks_unchanged=high,independent_model_checkpoints=model_count,independent_final_fusion_replay=True,
        nonprediction_bytes_exact=True,archive=backup,all_archive_member_hashes_verified=True,dependencies_verified=True,
        official_score=None,target_achieved=False,target_score=52,automatically_uploaded=False,github_push=False,server_shutdown=False)
    save(ROOT/'docs/v130_local_delivery.json',delivery);save(dest/'v130_delivery.json',delivery)
    sp=ROOT/'CURRENT_STATUS.json';state=load(sp);state.update(latest_completed_model_branch='v130',latest_completed_model_branch_is_proven_best=False,
        latest_aggregate_plan='v130',aggregate_status=delivery['status'],active_experiment='v130',active_experiment_status=delivery['status'],
        recommended_submission_version='v130',new_submission_created=True,running_new_training_processes=0,running_new_iteration_processes_at_completion=0,
        priority_next_step='Measure v130 low-task changes officially; keep v124 fallback and immutable high score tasks. Continue matched-baseline low-task development.')
    state['active_iteration']=dict(version='v130',baseline='v124',status=delivery['status'],official_score=None,high_score_tasks_immutable=True,running_training_processes=0)
    state['releases']['130']=delivery;state['v130_result']=delivery
    for x in audit['experiments']:state[x['version']+'_experiment_result']=dict(status='COMPLETE',accepted=x['accepted'],official_score=None)
    save(sp,state)
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8-sig');head='# 当前官方最佳 v124（51.115）；v130 低分任务新模型待评测\n\n'
    head+=f"v130 改变 {len(changed)} 项、{sum(changed.values())} 条预测（{', '.join(changed)}），其他 {22-len(changed)} 项及全部高分任务保留 v124 原预测。新模型按逐任务基线开发确认，失败候选未合入。模型重载、最终融合、归档哈希与模板非预测字节校验通过。\n\n"
    head+='尚无 v130 官方成绩，不能承诺到 52。[实验记录](docs/ITERATION_V130_20261027.md) · [本地交付](docs/v130_local_delivery.json)\n\n以下保留历史记录，当前状态以本段和 CURRENT_STATUS.json 为准。\n\n'
    if not text.startswith(head):p.write_text(head+text,encoding='utf-8')
    print(json.dumps(delivery,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
