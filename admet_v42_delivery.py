"""Describe final model evidence and preserve the scored baseline alongside the candidate."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    out = ROOT / 'admet_model_outputs_v42'
    dest = ROOT / 'admet_submissions/v42'
    file = next(dest.glob('*.csv'))
    verified = read(dest / 'local_verification.json')
    assert verified['status'] == 'INDEPENDENT_SOURCE_FUSION_AND_CSV_PASSED'
    report = read(out / 'analysis.json')
    uncertainty = read(out / 'uncertainty.json')
    neural = read(ROOT / 'admet_model_outputs_v40/analysis.json')
    parent = read(ROOT / 'admet_model_outputs_v41/analysis.json')
    paudit = read(ROOT / 'admet_model_outputs_v41/features/parents_audit.json')
    assert report['signature'] == uncertainty['signature']
    assert report['status'] == neural['status'] == parent['status'] == 'EVALUATED'
    selected = [r for r in report['tasks'] if r['selection']]
    assert selected
    neural_names = ['纯结构神经网络', '实验属性条件神经网络', '双头神经网络等权组合']
    parent_names = ['原始与去盐主体结构', '主体结构加官方实验邻域', '两主体分支等权组合']
    lines = ['# v42 神经结构表示与去盐主体视图实验', '',
        '平台基线为用户报告的v39：50.490698，较v36提高0.019738。v42尚未平台评测。', '',
        'v40学习完整的扩展分子结构视图，以门控融合Morgan2/FCFP3、Morgan3、路径/原子对/扭转、描述符/ChemBERTa四类输入，使用两层残差网络、纯结构预测头与FiLM实验属性条件预测头。所有描述符和辅助属性预处理只拟合在当前训练行。', '',
        '每个任务两个种子，90周期，固定第70/90周期快照。分类使用不加正类权重的BCE和0.05成对排序，MAE使用L1，Spearman使用拟合标签内高斯秩目标smooth-L1和0.10成对排序。预测头单独筛选；纯结构头推理对辅助属性不敏感，已验证。', '',
        'v41补充去盐主体分子视图。RDKit Cleanup、FragmentParent和Uncharger生成主体结构，提取主体的指纹、描述符以及原始/主体描述符差和片段/电荷差。原始结构和电荷信息仍作为输入，去盐身份不用于查找标签、聚合实验值或身份匹配。对指纹的方差选择只在拟合分子内进行。使用两个XGBoost种子与深度4/6，各650/850棵树，分结构与实验属性条件两个分支。', '',
        f"主体结构变化的分子数：{paudit['changed_molecules']:,}/{paudit['total_molecules']:,}（{paudit['changed_fraction']:.1%}）。标准化回退到原始分子的次数：{paudit['failed_standardizations']}。这些处理完全不读取标签。", '',
        f"v40独立筛选入选{len(neural['selected_tasks'])}个任务，v41入选{len(parent['selected_tasks'])}个任务。v42比较各源的冻结赢家及有重叠时的等权组合，最终更新{len(selected)}个任务。", '',
        '每个源均先冻结历史valid入围名单，再计算三折骨架代理。融合选择目标为0.45 valid增量+0.55 CV平均增量-0.04 CV标准差；要求valid>0.0002、CV均值>0、目标>0.0005。允许单折回退，权重只在预设网格中选择。', '',
        '全部GPU训练、重载和模型验证在Qwen服务器 czp123@100.109.143.42:/home/czp123/admet_v23_20260928 完成。CPU线程限制3；本机仅编辑源码、整理轻量预测与检查CSV。没有本机GPU训练或自动上传比赛。没有外部实验标签或测试目标标签。', '',
        '## 入选结果', '',
        'MAE增量为误差相对降低，其余为指标绝对增加，均不是平台分数。', '',
        '| 任务 | 指标 | 来源 | 新分支权重 | 完整v39历史valid增量 | 三折代理平均增量 |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for r in selected:
        s = r['selection']
        fmt = (lambda a: f'{a*100:+.3f}%') if r['metric'] == 'MAE' else (lambda a: f'{a:+.6f}')
        names = [f"v{a['version']}:{(neural_names if a['version'] == 40 else parent_names)[a['variant']]}" for a in s['sources']]
        lines.append(f"| {r['task']} | {r['metric']} | {' + '.join(names)} | {s['weight']:.2f} | {fmt(s['valid_gain'])} | {fmt(s['cv_mean'])} |")
    lines += ['', '## 验证边界与选型后诊断', '',
        '官方valid和骨架折已被多轮开发复用，存在选择偏差。CV比较对象是v39较新分支叠加历史v24代理，完整历史集成OOF没有重训。官方其他任务属性可能覆盖查询骨架，因此是条件预测，不能称为整个分子完全未知时的验证。去盐主体可能跨原始骨架组重合，已在元数据中记录，但未用主体身份匹配标签。', '',
        '以下300次复用valid骨架重采样是选型后诊断，不修正搜索偏差；留一折诊断也不属于完整嵌套验证。', '',
        '| 任务 | valid增量95%区间 | 重采样正收益占比 | 留一折选型平均增量 |',
        '| --- | --- | ---: | ---: |']
    for r in uncertainty['tasks']:
        lo, hi = r['post_selection_scaffold_bootstrap_95']
        fmt = (lambda a: f'{a*100:+.3f}%') if r['metric'] == 'MAE' else (lambda a: f'{a:+.6f}')
        interval = '单骨架，区间退化无效' if r.get('cluster_bootstrap_degenerate') else f'[{fmt(lo)}, {fmt(hi)}]'
        positive = '无法估计' if r.get('cluster_bootstrap_degenerate') else f"{r['bootstrap_positive_fraction']:.1%}"
        lines.append(f"| {r['task']} | {interval} | {positive} | {fmt(r['held_fold_mean'])} |")
    lines += ['', '各任务验证集的独立骨架簇数和最大簇行数已记录在uncertainty.json。单骨架集合的骨架重采样只能得到相同样本，不能用于估计不确定性。']
    lines += ['', '入选选项中的负收益折：', '']
    negatives = [(r['task'], j+1, g) for r in selected for j, g in enumerate(r['selection']['cv_gains']) if g < 0]
    lines += [f'- {task} 第{fold}折：{gain:+.6f}' for task, fold, gain in negatives] or ['无。']
    lines += ['',
        '官方总分公式、权重和分项得分未知，不能预测或保证平台涨分。最后报告第一名52.723518，超过其一分的目标53.723518尚未达到。v39保留为最后确认成绩的回退文件。', '',
        '## 交付核验', '', f'`{file.relative_to(ROOT).as_posix()}`', '',
        f"16,379行、22任务、6列，{verified['bytes']:,}字节。UTF-8 BOM、CRLF；所有非预测字段与官方模板逐行一致，键唯一、预测有限，分类范围[0,1]。", '',
        '远程入选快照和树模型全部重载重算；查询标签及目标任务辅助属性污染不改变输入。神经网络保存的预处理器已核对。本机用标准库独立重放源预测融合，核对源代码哈希与CSV哈希，通过。', '',
        f"SHA256：`{verified['sha256']}`。", '',
        '本轮只交付v42一个候选；v40/v41为内部源实验。']
    path = ROOT / 'admet_diagnostics/v42_report.md'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    state_path = ROOT / 'admet_current_candidate.json'
    state = read(state_path)
    assert state['platform_anchor_release'] == 'v39' and state['platform_anchor_score'] == 50.490698
    state.update(experimental_release='v42', experimental_submission=str(file), experimental_sha256=verified['sha256'],
        experimental_status='EXPERIMENTAL_AWAITING_PLATFORM', candidate_platform_score=None,
        awaiting='User platform evaluation of verified v42', report='admet_diagnostics/v42_report.md',
        verification='admet_submissions/v42/local_verification.json', remote_model_verification='admet_model_outputs_v42/model_replay.json',
        ready_for_platform_evaluation=True, changed_task_count=len(selected), automatically_uploaded=False,
        local_gpu_training=False, target_achieved=False, completed_experiments=['v40', 'v41', 'v42'])
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(file))
    print(str(path))


if __name__ == '__main__':
    main()
