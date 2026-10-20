"""Publish one verified candidate and describe selection uncertainty plainly."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    out = ROOT / 'admet_model_outputs_v40'
    report = read(out / 'analysis.json')
    uncertainty = read(out / 'uncertainty.json')
    screen = read(out / 'screen.json')
    assert report['status'] == 'EVALUATED' and screen['status'] == 'FROZEN'
    assert report['signature'] == uncertainty['signature'] == screen['signature']
    assert len(screen['tasks']) == 22
    selected = [r for r in report['tasks'] if r['selection']]
    names = ['纯结构神经网络', '实验属性条件神经网络', '两个神经预测头等权组合']
    lines = ['# v40 多视图分子神经网络与实验属性条件预测', '',
        '平台基线为用户报告的 v39：50.490698，比 v36 提高 0.019738。v40 尚未平台评测。', '',
        'v39 平台收益来自上一轮只改动 CYP2D6 底物的提交。本轮将完整扩展结构特征送入新的神经网络：Morgan2/FCFP3、半径3 Morgan、路径/原子对/扭转、描述符/ChemBERTa 共四个视图，各自经过投影和门控融合，并用两层残差网络建模。未将指纹按方差缩减为树模型的特征子集。', '',
        '网络有两个预测头。纯结构头只使用结构和预训练表示；条件头加入官方其他任务的同分子属性、相似分子属性以及自身任务训练邻域，并通过 FiLM 调制结构表示。条件输入只在当前拟合行内归一化，并在训练时按分子整通道随机丢弃20%。纯结构头推理不受辅助属性变化影响，已独立测试。', '',
        '每个任务使用两个种子、90个训练周期、批量256、隐藏维192、AdamW与余弦学习率。第70/90周期固定保存快照并平均预测。分类使用不加正类权重的BCE加0.05成对排序；MAE使用L1；Spearman使用拟合标签内高斯秩目标的smooth-L1加0.10成对排序。没有使用官方valid提前停止。', '',
        '先完成22项任务的历史valid初筛，冻结候选名单，再使用三折骨架代理选择变体和融合权重。目标函数为0.45 valid增量+0.55三折平均增量-0.04三折标准差；入选要求valid增量>0.0002、三折均值>0和目标函数>0.0005。允许单个任务的单折指标下降。', '',
        '全部训练与模型重放在Qwen远程服务器 czp123@100.109.143.42:/home/czp123/admet_v23_20260928 进行。线程限制3，本机没有GPU训练，仅编辑源代码和处理轻量预测证据。没有自动上传比赛。未使用外部实验标签或测试目标标签。', '',
        f'## 入选结果：{len(selected)}项', '',
        'MAE增量表示误差相对降低，其余指标为绝对增量。所有数值是验证指标，不是官方总分。', '',
        '| 任务 | 指标 | 变体 | 新分支权重 | 完整v39历史valid增量 | 三折代理平均增量 |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for r in selected:
        s = r['selection']
        fmt = (lambda a: f'{a*100:+.3f}%') if r['metric'] == 'MAE' else (lambda a: f'{a:+.6f}')
        lines.append(f"| {r['task']} | {r['metric']} | {names[s['variant']]} | {s['weight']:.2f} | {fmt(s['valid_gain'])} | {fmt(s['cv_mean'])} |")
    lines += ['', '## 不确定性', '',
        '历史valid和骨架折被多轮开发重复使用，存在选择偏差。CV只重建v39较新分支在历史v24代理上的融合，未重新训练完整历史集成OOF。其他任务的已知属性可能覆盖查询骨架，属于条件预测。以下300次骨架重采样是选型后诊断，不会消除模型搜索偏差；留一折诊断也不是完整嵌套验证。', '',
        '| 任务 | valid增量95%区间 | 重采样正收益占比 | 留一折选型平均增量 |',
        '| --- | --- | ---: | ---: |']
    for r in uncertainty['tasks']:
        lo, hi = r['post_selection_scaffold_bootstrap_95']
        fmt = (lambda a: f'{a*100:+.3f}%') if r['metric'] == 'MAE' else (lambda a: f'{a:+.6f}')
        lines.append(f"| {r['task']} | [{fmt(lo)}, {fmt(hi)}] | {r['bootstrap_positive_fraction']:.1%} | {fmt(r['held_fold_mean'])} |")
    lines += ['', '入选选项中的负收益折：', '']
    negatives = [(r['task'], j+1, g) for r in selected for j, g in enumerate(r['selection']['cv_gains']) if g < 0]
    lines += [f'- {task} 第{fold}折：{gain:+.6f}' for task, fold, gain in negatives] or ['无。']
    lines += ['', '官方总分公式、任务权重和分项得分未知，没有依据保证涨分或指定涨幅。最后报告的第一名为52.723518，超过其一分的目标为53.723518，尚未达到。v39保留为最后确认得分的回退文件。', '']
    if selected:
        dest = ROOT / 'admet_submissions/v40'
        file = next(dest.glob('*.csv'))
        verified = read(dest / 'local_verification.json')
        assert verified['status'] == 'INDEPENDENT_NEURAL_FUSION_AND_CSV_PASSED'
        lines += ['## 交付核验', '', f'`{file.relative_to(ROOT).as_posix()}`', '',
            f"16,379行、22任务、6列，{verified['bytes']:,}字节。UTF-8 BOM、CRLF；所有身份字段和行序逐行与官方模板一致，键唯一、数值有限、分类范围为[0,1]。", '',
            '远程重新构建特征、核对保存的预处理器、重载全部入选快照并重算最终预测，通过。污染测试标签及目标任务辅助属性块不影响特征。本机用标准库独立重放最终融合，并核对CSV、源代码哈希与远程模型记录，通过。', '',
            f"SHA256：`{verified['sha256']}`。"]
    else:
        lines += ['本轮没有入选候选，没有导出伪装成新版本的提交文件。']
    path = ROOT / 'admet_diagnostics/v40_report.md'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    state_path = ROOT / 'admet_current_candidate.json'
    state = read(state_path)
    assert state['platform_anchor_release'] == 'v39' and state['platform_anchor_score'] == 50.490698
    state.update(experimental_release='v40', report='admet_diagnostics/v40_report.md',
        changed_task_count=len(selected), automatically_uploaded=False, local_gpu_training=False, target_achieved=False)
    if selected:
        state.update(experimental_submission=str(file), experimental_sha256=verified['sha256'],
            experimental_status='EXPERIMENTAL_AWAITING_PLATFORM', candidate_platform_score=None,
            awaiting='User platform evaluation of verified v40', ready_for_platform_evaluation=True,
            verification='admet_submissions/v40/local_verification.json',
            remote_model_verification='admet_model_outputs_v40/model_replay.json')
    else:
        state.update(experimental_status='NO_CANDIDATE', awaiting='New model experiment needed', ready_for_platform_evaluation=False)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(path))


if __name__ == '__main__':
    main()
