"""Write final selected-task evidence directly from saved analysis."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    analysis=json.loads((ROOT/'admet_model_outputs_v32/analysis.json').read_text(encoding='utf-8'))
    selected=[r for r in analysis['tasks'] if r['selection']]
    names=['conditional neural','embedding conditional boost','descriptor conditional boost','neural + descriptor boost','neural + embedding boost']
    lines=['# v32 模型改进与提交核验','',
        '基线为用户报告得分 50.386293 的 v29，较 v24 增加 0.137534。本轮对完整 v29 历史 valid 预测检查增量，并允许个别验证折回退。官方分项成绩及总分公式未知，不能预测新文件的实际得分。','',
        '训练全部运行于 Qwen 服务器 `czp123@100.109.143.42:/home/czp123/admet_v23_20260928`，本机没有 GPU 训练。','',
        '新增联合指纹、描述符、ChemBERTa、其他任务官方实验属性的条件神经网络，加入条件门控和 15% 辅助属性随机遮蔽；神经网络采用两个种子与两个训练快照。条件 CatBoost 采用两个种子、两种树深度。三类原始候选和两个互补混合候选按任务筛选，未通过的任务使用 v29 原预测。','',
        '## 选择结果','',
        f"入选 {len(selected)} 个任务。以下是对完整 v29 历史 valid 的指标变化及对 v29 新分支 + v24 神经分支代理 CV 的平均变化。MAE 为相对下降，其他指标为绝对增加；不能换算成比赛分数。",'',
        '| 任务 | 指标 | 新候选 | 权重 | valid 改善 | 三折平均代理改善 |','| --- | --- | --- | ---: | ---: | ---: |']
    for r in selected:
        s=r['selection'];v=s['valid_gain'];c=s['cv_mean']
        vv=f'{100*v:.3f}%' if r['metric']=='MAE' else f'{v:+.6f}'
        cc=f'{100*c:.3f}%' if r['metric']=='MAE' else f'{c:+.6f}'
        lines.append(f"| {r['task']} | {r['metric']} | {names[s['variant']]} | {s['weight']:.2f} | {vv} | {cc} |")
    lines+=['','## 验证局限','',
        '历史 valid 与训练骨架折已在多轮开发中复用，存在选择偏差。CV 不能重放完整历史树模型集成；它重放 v29 新分支和原 v24 神经分支，属于代理比较。条件模型可以使用目标分子在其他任务中已公开的实验属性，不代表完全未知分子的骨架泛化。','',
        '所有输入属性仅来自其他任务的官方 train（验证/CV）或 train+valid（最终），完整移除当前目标属性及观测掩码，未使用测试标签或外部标签。四个目标任务的目标/测试标签扰动隔离审计及网络批次不变性检查通过。','',
        '选择规则：valid 增量 >0.0001、三折平均代理增量为正、至少两折非负。仍可能存在单折或官方单项回退。',
        '']
    negatives=[(r['task'],j,g) for r in selected for j,g in enumerate(r['selection']['cv_gains']) if g<0]
    if negatives:
        lines+=['入选选项中的负收益折：']
        lines += [f'- {t} 第 {j} 折：{g:+.6f}' for t,j,g in negatives]
        lines+=['']
    nested_negative=[(r['task'],sum(r['held_fold_cv_selection_gains'])/3) for r in selected if sum(r['held_fold_cv_selection_gains'])<0]
    if nested_negative:
        lines+=['补充的留一折选择诊断：用另两折选择候选和权重，再看留出折。以下任务的平均结果为负，说明即使固定选项三折平均为正，选择过程仍可能不稳定。该诊断没有用于事后修改预先设定的选择规则；这些任务保留为探索候选。','']
        lines += [f'- {t}：平均留出折代理增量 {g:+.6f}' for t,g in nested_negative]
        lines+=['']
    dest=ROOT/'admet_submissions/v32'
    if dest.exists():
        file=next(dest.glob('*.csv'));info=json.loads((dest/'verification.json').read_text(encoding='utf-8'))
        replay=json.loads((ROOT/'admet_model_outputs_v32/model_replay.json').read_text(encoding='utf-8'))
        local=json.loads((dest/'local_verification.json').read_text(encoding='utf-8'))
        assert replay['status']=='FINAL_MODEL_REPLAY_PASSED' and local['sha256']==info['sha256']
        lines+=['## 提交文件','',f'`{file.relative_to(ROOT).as_posix()}`','',f"SHA256：`{info['sha256']}`。",'',
            f"文件大小 {info['bytes']:,} 字节，16,379 条数据、22 个任务、6 列、UTF-8 BOM、CRLF。身份字段与官方模板逐行一致。",'',
            '全部最终模型已在远程重新加载并重算核验。本机使用标准 Python 独立重放候选混合和 CSV 融合，并核验源码哈希、提交哈希、行列、模板、编码及预测范围，检查通过。','',
            '未自动上传平台。v32 实际分数未知，v29 继续作为最后确认得分的版本。','']
        state_path=ROOT/'admet_current_candidate.json';state=json.loads(state_path.read_text(encoding='utf-8'))
        state.update(experimental_release='v32',experimental_status='EXPERIMENTAL_AWAITING_PLATFORM',experimental_submission=str(file),
            experimental_sha256=info['sha256'],ready_for_platform_evaluation=True,changed_task_count=len(selected),
            report='admet_diagnostics/v32_report.md',verification='admet_submissions/v32/local_verification.json',
            remote_model_verification='admet_model_outputs_v32/model_replay.json',awaiting='User platform evaluation of v32',candidate_platform_score=None)
        state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'admet_diagnostics/v32_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(selected_tasks=[r['task'] for r in selected],official_score_prediction=None),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
