import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    dest=ROOT/'admet_submissions/v36';file=next(dest.glob('*.csv'));verify=json.loads((dest/'local_verification.json').read_text(encoding='utf-8'))
    out=ROOT/'admet_model_outputs_v36';r=json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    assert verify['status']=='INDEPENDENT_MODEL_AND_CSV_REPLAY_PASSED'
    names={33:'邻域实验信息神经网络/CatBoost',34:'邻域实验信息核模型',35:'邻域实验信息 XGBoost'}
    lines=['# v36 邻域实验属性迁移与互补模型集成','',
        '最新平台基线为用户报告的 v32：50.399027，比 v29 增加 0.012734。v36 尚未平台评测，不提供官方分数预测。','',
        '本轮把“同分子的其他已知实验属性”扩展到“结构相似分子的官方实验属性”：检索 Morgan/FCFP 化学邻居，加入实验值聚合、相似度、标签分散度和有效邻居数量，再由神经网络、CatBoost、核模型和 XGBoost 学习结构与 ADMET 之间的关系。按任务筛选分支，并检验各分支的等权组合。','',
        '全部训练在 Qwen 远程服务器 `czp123@100.109.143.42:/home/czp123/admet_v23_20260928` 完成。本机仅编辑源码、下载轻量预测证据和核验 CSV，没有 GPU 训练。没有自动提交比赛平台。','',
        'v33 高耗时源完成 12 个任务的筛选，入选 6 项，其余未完成候选排除。v34 核模型覆盖训练行数不超过 3,000 的任务。v35 XGBoost 覆盖全部 22 项。计算资源从未完成的慢源搜索转向全任务快源和最终重训，此调整没有读取测试标签。','',
        '自身任务邻域只使用当前拟合行标签，并排除同骨架分子；其他任务只使用官方 train（验证/CV）或 train+valid（最终）标签，目标任务的辅助特征块完整移除。重复身份标签先聚合。未使用外部实验标签或测试目标标签。','',
        f"最终更新 {len(r['selected_tasks'])} 个任务。以下 MAE 为误差相对减少，其他为指标绝对增加；允许单折回退。",'',
        '| 任务 | 指标 | 来源 | 新分支权重 | 完整 v32 历史 valid 增量 | 三折代理平均增量 |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    selected=[x for x in r['tasks'] if x['selection']]
    for x in selected:
        s=x['selection'];a,c=s['valid_gain'],s['cv_mean'];mae=x['metric']=='MAE'
        a=f'{a*100:.3f}%' if mae else f'{a:+.6f}';c=f'{c*100:.3f}%' if mae else f'{c:+.6f}'
        source=' + '.join(f"v{v['version']}" for v in s['sources'])
        lines.append(f"| {x['task']} | {x['metric']} | {source} | {s['weight']:.2f} | {a} | {c} |")
    lines+=['','## 验证边界与风险','',
        '完整历史 valid 重放 v32，但 valid 和骨架折已经多轮开发复用，存在选择偏差。CV 基线仍是 v32 新分支在 v24 神经分支上的代理融合，没有完整历史集成 OOF。其他任务的公开属性可能覆盖查询骨架，因此属于有其他属性信息的条件预测。多个候选的组合搜索进一步增加选择偏差。','',
        '官方任务权重与总分公式未知，没有任何依据保证达到 51 分或某个指定涨幅。历史验证正收益仍可能在官方盲测上下降。当前目标为超过最后报告的第一名 52.723518 至少一分，即 53.723518，尚未达到。','',
        '入选选项的负收益折：','']
    negatives=0
    for x in selected:
        for j,g in enumerate(x['selection']['cv_gains']):
            if g<0:lines.append(f"- {x['task']} 第 {j+1} 折：{g:+.6f}");negatives+=1
    if not negatives:lines.append('无。')
    uncertainty=out/'uncertainty.json'
    if uncertainty.exists():
        u=json.loads(uncertainty.read_text(encoding='utf-8'))
        assert u['signature']==r['signature']
        lines+=['','## 选型后的不确定性','',
            '以下 95% 区间在复用 valid 上按骨架重采样 400 次，属于选型后诊断，没有消除模型搜索偏差。留一折诊断用另两折选择已有源候选，再计算留出折增量；源候选本身已被搜索，因此仍不是真正完整嵌套验证。此诊断不改变已冻结的提交选择。','',
            '| 任务 | valid 增量 95% 区间 | 重采样正收益占比 | 留一折选型平均增量 |',
            '| --- | --- | ---: | ---: |']
        for x in u['tasks']:
            lo,hi=x['post_selection_scaffold_bootstrap_95']
            lines.append(f"| {x['task']} | [{lo:+.6f}, {hi:+.6f}] | {x['bootstrap_positive_fraction']:.1%} | {x['held_fold_mean']:+.6f} |")
    lines+=['','## 交付与核验','',f'`{file.relative_to(ROOT).as_posix()}`','',
        f"16,379 行、22 个任务、6 列，{verify['bytes']:,} 字节，UTF-8 BOM、CRLF。所有身份字段逐行与官方模板一致。远程所有入选模型已重载重算；本机独立重放融合，检查源代码哈希、CSV 哈希、数值范围、编码、唯一键和排序变化，通过。",'',
        f"SHA256：`{verify['sha256']}`。",'',
        'v32 原文件继续保留为最后确认得分的基线；v36 是待平台评测候选。']
    report=ROOT/'admet_diagnostics/v36_report.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    p=ROOT/'admet_current_candidate.json';state=json.loads(p.read_text(encoding='utf-8'))
    assert state['platform_anchor_release']=='v32' and state['platform_anchor_score']==50.399027
    state.update(experimental_release='v36',experimental_submission=str(file),experimental_sha256=verify['sha256'],
        experimental_status='EXPERIMENTAL_AWAITING_PLATFORM',candidate_platform_score=None,awaiting='User platform evaluation of v36',
        report='admet_diagnostics/v36_report.md',verification='admet_submissions/v36/local_verification.json',
        remote_model_verification='admet_model_outputs_v36/model_replay.json',ready_for_platform_evaluation=True,
        changed_task_count=len(selected),automatically_uploaded=False,local_gpu_training=False,target_achieved=False)
    p.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(file));print(str(report))
if __name__=='__main__':main()
