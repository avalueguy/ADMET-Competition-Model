"""Record the completed experiment, its rejected branches and the verified candidate."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    dest = ROOT / 'admet_submissions/v39'
    file = next(dest.glob('*.csv'))
    verified = read(dest / 'local_verification.json')
    assert verified['status'] == 'INDEPENDENT_SOURCE_FUSION_AND_CSV_PASSED'
    out = ROOT / 'admet_model_outputs_v39'
    analysis = read(out / 'analysis.json')
    uncertainty = read(out / 'uncertainty.json')
    screen = read(ROOT / 'admet_model_outputs_v37/screen.json')
    kernel = read(ROOT / 'admet_model_outputs_v38/analysis.json')
    tree = read(ROOT / 'admet_model_outputs_v37/analysis.json')
    assert analysis['status'] == kernel['status'] == tree['status'] == 'EVALUATED'
    assert screen['status'] == 'FROZEN' and len(screen['tasks']) == 22
    assert analysis['signature'] == uncertainty['signature']
    selected = [row for row in analysis['tasks'] if row['selection']]
    assert selected and not kernel['selected_tasks']
    names = {0: '扩展结构常规损失', 1: '扩展结构与官方实验邻域', 2: '指标对应损失'}
    lines = ['# v39：v37/v38 实验完成与交付记录', '',
             '整理日期：2026-10-08。最新平台基线为用户报告的 v36：50.470960。v39 尚未平台评测。', '',
             '本轮远程训练已完成。v37 对全部 22 个任务筛选扩展化学特征和不同训练损失，只保留 CYP2D6 底物任务；v38 对 9 个连续属性任务测试稳健核模型，全部未达到入选要求。v39 按冻结的选择规则汇总候选，因此最终文件内容与 v37 完全相同。v38 的完成不是新增收益，v39 也没有额外模型增益。', '',
             'v37 新增半径 3 Morgan 计数指纹（4,096 维）、路径指纹（2,048 维）、原子对和扭转指纹（各 1,024 维）。每个视图仅在当前拟合分子内按方差选择 384 列，总计 1,536 列；描述符和预训练表示同样只在拟合行内归一化。测试了结构分支、结构加其他任务官方属性分支、指标对应损失分支。两个随机种子，树深度 4/6，各 650/850 棵树。', '',
             '指标对应损失包括 AUPRC 的 rank:map、AUROC/Spearman 的 rank:pairwise 和 MAE 的中位数分位损失。这些损失分支没有进入最终文件。最终采用常规损失的结构与实验邻域互补模型，不能把未入选方法描述为已实现的评分提升。', '',
             'v38 使用六种广义 Tanimoto 结构相似度以及描述符/ChemBERTa RBF 核，并针对 MAE 采用迭代重加权稳健拟合。对 5 个 MAE 和 4 个 Spearman 任务的筛选均不及 v36，全部排除。', '',
             '训练与模型重放全部在 Qwen 远程服务器 czp123@100.109.143.42:/home/czp123/admet_v23_20260928 完成，线程限制为 3。本机只编辑代码、整理小型预测证据和验证格式，没有 GPU 训练。没有自动提交比赛平台。', '',
             '## 最终预测变化', '',
             '| 任务 | 指标 | 新分支权重 | 完整 v36 历史 valid | v39 历史 valid | 三折代理增量 |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    bytask = {row['task']: row for row in uncertainty['tasks']}
    for row in selected:
        s = row['selection']
        u = bytask[row['task']]
        lines.append(f"| {row['task']} | {row['metric']} | {s['weight']:.2f} | {u['valid_before']:.6f} | {u['valid_after']:.6f} | {s['cv_mean']:+.6f} |")
    lines += ['',
              'CYP2D6 底物任务的新分支由扩展结构常规损失模型与结构加实验邻域模型等权组合，再以 0.35 权重和 v36 融合。135 行预测改变，其中 119 行排序改变；其余 21 个任务逐行保持 v36 原值。', '',
              '三折 AUPRC 增量分别为 +0.010141、+0.046710、+0.049691。完整历史 valid 增量仅 +0.000379。两者差距明显，说明候选的泛化收益仍有较大不确定性。', '',
              '## 验证边界', '',
              'valid 和骨架折被多轮开发重复使用，存在选择偏差。CV 的比较对象是 v36 新分支叠加历史 v24 代理，并非完整历史集成 OOF。其他任务提供的属性可以覆盖查询骨架，因此是条件预测。训练没有使用外部实验标签或测试目标标签。', '',
              '| 任务 | 选型后 valid 增量 95% 区间 | 骨架重采样正收益占比 | 留一折选型平均增量 |',
              '| --- | --- | ---: | ---: |']
    for row in uncertainty['tasks']:
        lo, hi = row['post_selection_scaffold_bootstrap_95']
        lines.append(f"| {row['task']} | [{lo:+.6f}, {hi:+.6f}] | {row['bootstrap_positive_fraction']:.1%} | {row['held_fold_mean']:+.6f} |")
    lines += ['',
              '区间基于 300 次复用 valid 的骨架重采样，跨过零，且正收益占比只有 40.7%；它不消除模型搜索偏差。留一折诊断也不是完整嵌套验证。不能把 CV 增量当成官方平台分数或独立收益估计。', '',
              '官方总分公式、任务权重和分项得分未获得，无法保证本版本涨分或指定涨幅。本轮不构成已证实的大幅突破。用户最后报告第一名 52.723518，超过其一分的目标为 53.723518，尚未达到。v36 继续作为最后确认得分的回退版本。', '',
              '## 核验与提交', '', f'`{file.relative_to(ROOT).as_posix()}`', '',
              f"16,379 行、22 个任务、6 列，{verified['bytes']:,} 字节。UTF-8 BOM、CRLF，列名、任务顺序、样本索引、drug_id、SMILES 和 metric 与官方模板逐行一致，无重复键，预测值有限，分类值在 [0,1]。", '',
              '远程模型已重载并重算。污染目标任务辅助属性块和查询标签没有改变输入特征；特征选择仅使用拟合分子。本机用标准库独立重放最终融合，并核对 CSV、代码哈希与模型核验记录，通过。', '',
              '训练中一次模型重载的 CPU/GPU 预测路径产生约 3.8e-6 的浮点差异；恢复程序明确将重载预测设为 CUDA，保持原重放容差，复用已完成缓存后通过核验。该运行时修正已记录并核对源码哈希。', '',
              f"SHA256：`{verified['sha256']}`。", '',
              '本轮仅交付 v39 候选。v37 是相同内容的内部产物，不需要重复上传。']
    report = ROOT / 'admet_diagnostics/v39_report.md'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    path = ROOT / 'admet_current_candidate.json'
    state = read(path)
    assert state['platform_anchor_release'] == 'v36' and state['platform_anchor_score'] == 50.470960
    state.update(experimental_release='v39', experimental_submission=str(file),
                 experimental_sha256=verified['sha256'], experimental_status='EXPERIMENTAL_AWAITING_PLATFORM',
                 candidate_platform_score=None, awaiting='User platform evaluation of verified v39',
                 report='admet_diagnostics/v39_report.md', verification='admet_submissions/v39/local_verification.json',
                 remote_model_verification='admet_model_outputs_v39/model_replay.json',
                 ready_for_platform_evaluation=True, changed_task_count=len(selected),
                 automatically_uploaded=False, local_gpu_training=False, target_achieved=False,
                 completed_experiments=['v37', 'v38', 'v39'], equivalent_internal_candidate='v37',
                 delivered_on='2026-10-08')
    state.pop('remote_run_pid', None)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(file))
    print(str(report))


if __name__ == '__main__':
    main()
