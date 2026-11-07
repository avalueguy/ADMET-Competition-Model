"""Verify final fusion independently using stdlib, and deliver v144 CSV."""
import json,shutil,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from finalize_v124_local import load,save,sha,rows,array,ranks
from verify_submission import verify
from repack_upload_template import field_spans
def main():
    report=load(ROOT/'admet_model_outputs_v144/evidence.json');audit=load(ROOT/'admet_model_outputs_v144/independent_verification.json')
    backup=load(ROOT/'releases/v144_broad_task_models_20261027.archive.json');local=load(ROOT/'docs/v144_local_archive_verification.json')
    assert audit['status']=='PASS' and local['all_member_hashes_verified'] and local['dependency_hashes_verified'] and backup['sha256']==local['archive_sha256']
    anchor=next((ROOT/'admet_submissions/v130_low_task_experts').glob('*.csv'));file=ROOT/report['file'];assert sha(anchor)==report['baseline_sha256'] and sha(file)==report['sha256']
    old,new=rows(anchor),rows(file);assert len(old)==len(new)==16380
    for task,ch in report['selected'].items():
        number=ch['version'];r=ch['record'];cfg=r['config'];folder=ROOT/f'admet_model_outputs_{number}/models/{task}/final'
        if number in ('v134','v139'):path=folder/(cfg['view']+'_C'+str(cfg['C'])+'.npz')
        elif number=='v137':path=folder/'predictions.npz'
        elif number=='v141':path=folder/(cfg['view']+'_leaf'+str(cfg['leaf']))/'predictions.npz'
        else:path=folder/(cfg['view']+'_d'+str(cfg['depth']))/'predictions.npz'
        p=array(path,'p');pos=[i for i,z in enumerate(old) if i and z[0]==task];a=[float(old[i][4]) for i in pos];w=r['weight'];assert len(p)==len(pos)
        value=[(1-w)*x+w*y for x,y in zip(a,p)] if r['metric']=='MAE' else [(1-w)*x/len(a)+w*y/len(a) for x,y in zip(ranks(a),ranks(p))]
        assert all(abs(float(new[i][4])-z)<1e-14 for i,z in zip(pos,value))
    changes=Counter()
    for a,z in zip(old[1:],new[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[0] not in report['selected']:assert a==z
        if a[4]!=z[4]:changes[a[0]]+=1
    assert dict(changes)==report['verification']['changed_tasks'] and 'hia_hou' not in changes
    template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template);raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);pieces=[];cursor=0
    for i in range(1,len(new)):
        lo,hi=ss[i][4];a,z=tt[i][4];pieces.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    pieces.append(raw[cursor:]);assert b''.join(pieces)==ref
    assert fmt['rows']==16379 and fmt['tasks']==22 and fmt['utf8_bom'] and fmt['crlf_only']
    fmt.update(nonprediction_bytes_exact=True,independent_final_fusion_replay=True,protected_task_predictions_exact=True);save(ROOT/'docs/v144_local_submission_verification.json',fmt)
    lines=['# v144：覆盖全部可提升任务的新模型实验','',
        '基线 v130 官方成绩 51.146，分项均值 51.145616。较 v124 提高 0.030526，来自 CYP3A4 底物与 hERG。用户授权高分任务同步优化；HIA 的 AUROC 已为 1，因此保留。v144 尚无官方成绩，未确认达到 52。','',
        '本轮比较七条路线：回归联合核 SVR、全结构/增强化学特征 GPU XGBoost、其他任务分子近邻测量、任务独立成对排序神经网络、分类联合核 SVC、有限深度 ExtraTrees、固定 CPU CatBoost。各路线先在 fold0/1 冻结任务配置和融合比例，再确认 fold2/valid。','',
        '同一任务的跨路线选择只比较开发折最小增益；即使开发胜者确认失败，也不转用其他路线。这避免根据确认成绩重新挑模型。全部替换任务均要求四分区正增益。部分历史基线 OOF 仍是代理，公开标签长期复用；本地结果不能直接换算官方总分。','',
        '| 任务 | 路线 | 融合比例 | 分区 | 基线 | 候选 | 指标增益 |','|---|---|---:|---|---:|---:|---:|']
    for task,ch in report['selected'].items():
        for r in ch['record']['confirmation']:lines.append(f"| {task} | {ch['version']} | {ch['record']['weight']:.2f} | {r['part']} | {r['baseline']:.6f} | {r['candidate']:.6f} | {r['gain']:+.6f} |")
    lines+=['','MAE 的增益表示相对误差减少；其余为 AUROC/AUPRC/Spearman 的绝对增益。','',
        '| 任务 | 确认部分 | 骨架 bootstrap 增益 95% 区间 |','|---|---|---|']
    group_audit=load(ROOT/'docs/v144_scaffold_group_diagnostics.json')
    group_counts={(r['task'],r['part']):r for r in group_audit['records']}
    for r in audit['checks']:
        if 'scaffold_bootstrap_gain_ci95' in r:
            lo,hi=r['scaffold_bootstrap_gain_ci95'];group=group_counts[(r['task'],r['part'])]
            interval=f'[{lo:+.6f}, {hi:+.6f}]' if group['bootstrap_interpretable'] else '不可估计：现有分组仅 1 组'
            lines.append(f"| {r['task']} | {r['part']} | {interval} |")
    lines+=['','这些区间不能消除历史标签复用和模型选择偏差。AMES valid 的现有骨架分组只有 1 组，重采样退化，不能将其等于点估计的区间解释为稳定证据。其余区间也多跨过零。','',
        '## 实验完成情况','']
    for r in report['experiments']:lines.append(f"- {r['version']} 四分区通过：{', '.join(r['accepted']) or '无'}。最终是否合入还取决于开发折跨路线胜者规则。")
    lines+=['','v132 因 GPU 与 CPU 推理存在浮点差异停止，v135 统一 CPU 推理后重训；v133 因大任务无深度限制森林耗时过长停止，改用 v141 限定深度版本重训。v142 因与其他 GPU 进程并行时耗时过高，在完成首分区后停止，改用 v143 CPU CatBoost；原始日志与部分产物保留。','',
        f"最终改变 {len(changes)} 项、{sum(changes.values())} 条预测：{dict(changes)}。其他 {22-len(changes)} 项与 v130 预测字符串完全一致。",'',
        '所有新增模型只使用官方 train/valid 标签和官方结构生成的特征；目标任务测量及其近邻汇总列被排除。未读取 test 标签，未新增外部数据或预训练权重。已有历史基线预训练来源资格未在本轮重审。','',
        '模型已独立重载、查询标签扰动及最终融合复算通过；CSV 的行顺序、Drug_ID 原字符串及全部非预测字节与官方模板一致。','',
        f"CSV SHA256：`{report['sha256']}`",'',
        '验证阶段将重复计算的骨架行索引改为一次缓存；核对索引序列完全一致，随机抽样、模型选择和预测均未改变。原始验证代码及中断日志保留。','',
        f"备份：`{backup['path']}`，{backup['files']} 个成员全部哈希验证，依赖包哈希验证通过。",'',
        '未自动上传平台、未推送 GitHub、未关闭服务器。新预测的官方成绩可能波动；仍需平台实测。']
    doc=ROOT/'docs/ITERATION_V144_20261027.md';doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dest=Path('D:/bili_download/v144_broad_task_models');dest.mkdir(parents=True,exist_ok=True);upload=dest/file.name
    if upload.exists():assert upload.read_bytes()==raw
    else:shutil.copy2(file,upload)
    assert sha(upload)==report['sha256'];shutil.copy2(file.with_suffix('.verification.json'),upload.with_suffix('.verification.json'));shutil.copy2(doc,dest/'v144_说明.md')
    delivery=dict(status='LOCAL_DELIVERY_COMPLETE_AWAITING_PLATFORM',version='v144',baseline='v130',baseline_official_score=51.146,csv=report['file'],upload_copy=str(upload),sha256=report['sha256'],
        changed_tasks=dict(changes),changed_predictions=sum(changes.values()),unchanged_tasks=22-len(changes),independent_final_fusion_replay=True,nonprediction_bytes_exact=True,
        archive=backup,all_archive_member_hashes_verified=True,dependencies_verified=True,official_score=None,target_achieved=False,target_score=52,automatically_uploaded=False,github_push=False,server_shutdown=False)
    save(ROOT/'docs/v144_local_delivery.json',delivery);save(dest/'v144_delivery.json',delivery)
    p=ROOT/'CURRENT_STATUS.json';state=load(p);state.update(latest_completed_model_branch='v144',latest_completed_model_branch_is_proven_best=False,latest_aggregate_plan='v144',aggregate_status=delivery['status'],
        active_experiment='v144',active_experiment_status=delivery['status'],recommended_submission_version='v144',new_submission_created=True,running_new_training_processes=0,
        best_official_version='v130',best_official_version_user_reported='v130',best_official_score_user_reported=51.146,
        last_user_reported_official_version='v130',last_user_reported_official_score=51.146,official_score=51.146,
        running_new_iteration_processes_at_completion=0,priority_next_step='Measure v144 officially and compare per-task changes against v130; preserve v130 fallback.')
    state['latest_user_reported_official_score']=dict(version='v130',score=51.146,previous_version='v124',previous_score=51.115,observed_gain=.031,
        source='User-pasted task scorecard associated with latest delivered v130; platform hash not exposed.',reported_date='2026-10-27',uploaded_byte_hash_confirmed=False,target_score=52.,gap_to_target=.854)
    state['active_iteration']=dict(version='v144',baseline='v130',status=delivery['status'],official_score=None,running_training_processes=0)
    state['releases']['144']=delivery;state['v144_result']=delivery
    for x in report['experiments']:state[x['version']+'_experiment_result']=dict(status='COMPLETE',accepted=x['accepted'],official_score=None)
    save(p,state);p=ROOT/'README.md';text=p.read_text(encoding='utf-8-sig');head=f"# 当前官方最佳 v130（51.146）；v144 广泛任务新模型待评测\n\nv144 完成所有 21 个可提升任务的多路线筛选，最终更新 {len(changes)} 项、{sum(changes.values())} 条预测；其他 {22-len(changes)} 项保持 v130。模型重载、融合、模板字节及备份哈希验证通过。官方成绩未知，尚未确认到 52。\n\n[实验记录](docs/ITERATION_V144_20261027.md) · [本地交付](docs/v144_local_delivery.json)\n\n历史记录如下。\n\n"
    if not text.startswith(head):p.write_text(head+text,encoding='utf-8')
    print(json.dumps(delivery,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
