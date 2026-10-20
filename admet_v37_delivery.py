"""Write a readable comparison report and expose one verified candidate."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def main():
    dest = ROOT / 'admet_submissions/v37'
    file = next(dest.glob('*.csv'))
    verify = json.loads((dest / 'local_verification.json').read_text(encoding='utf-8'))
    assert verify['status'] == 'INDEPENDENT_FUSION_AND_CSV_VERIFICATION_PASSED'
    out = ROOT / 'admet_model_outputs_v37'
    r = json.loads((out / 'analysis.json').read_text(encoding='utf-8'))
    screen = json.loads((out / 'screen.json').read_text(encoding='utf-8'))
    uncertainty = json.loads((out / 'uncertainty.json').read_text(encoding='utf-8'))
    assert r['signature'] == uncertainty['signature'] == screen['signature']
    selected = [x for x in r['tasks'] if x['selection']]
    names = {0: '扩展结构常规损失', 1: '扩展结构与实验邻域', 2: '实验邻域与指标对应损失'}
    lines = ['# v37 分子结构视图扩展与指标对应训练', '',
             '平台确认基线为用户报告的 v36：50.470960，较 v32 增加 0.071933。v37 尚未平台评测。', '',
             '本轮把所有 22 个任务与重建的完整 v36 历史 valid 结果比较，增加 4,096 维半径 3 Morgan 计数指纹、2,048 维路径指纹、1,024 维原子对和 1,024 维扭转指纹。每个视图在当前拟合分子内按方差选择 384 列，合计 1,536 个新结构列。描述符与预训练表示的归一化同样仅拟合于当前训练行。', '',
             '三个分支分别学习结构、结构加官方其他实验属性、结构加实验属性并使用对应损失。AUPRC 采用 XGBoost rank:map，AUROC 和 Spearman 采用 rank:pairwise，Spearman 相关性目标在拟合标签内转换为 32 档相关等级；MAE 分支采用中位数分位损失。常规损失分支提供比较与互补，并取消之前普遍使用的正类加权。指标对应损失不代表一定提升；只有实际入选分支进入提交。', '',
             '两个随机种子、深度 4/6，各 650/850 棵树。候选先在复用 valid 上冻结入围名单，再评价三折骨架代理；按 0.45 valid 增量 +0.55 CV 平均增量 -0.04 CV 标准差筛选。允许单折指标下降，关注综合收益。', '',
             '全部训练在 Qwen 远程服务器 czp123@100.109.143.42:/home/czp123/admet_v23_20260928 完成。线程限制为 3，未启动本机 GPU 训练，未自动提交比赛。没有使用外部实验标签或测试标签。', '',
             f'22 个任务完成筛选，最终更新 {len(selected)} 项。MAE 增量表示误差相对降低；其他指标为绝对增量。', '',
             '| 任务 | 指标 | 入选分支 | 新分支权重 | 完整 v36 历史 valid 增量 | 三折代理平均增量 |',
             '| --- | --- | --- | ---: | ---: | ---: |']
    for x in selected:
        s = x['selection']
        fmt = (lambda a: f'{a*100:+.3f}%') if x['metric'] == 'MAE' else (lambda a: f'{a:+.6f}')
        sources = ' + '.join(names[f] for f in s['families'])
        lines.append(f"| {x['task']} | {x['metric']} | {sources} | {s['weight']:.2f} | {fmt(s['valid_gain'])} | {fmt(s['cv_mean'])} |")
    lines += ['', '## 选型后诊断与边界', '',
              'valid 和骨架折被多轮重复开发使用，存在模型选择偏差。CV 比较对象是 v36 新分支叠加历史 v24 代理，未完整重训历史集成。其他任务提供的属性可以覆盖查询骨架，因此属于条件预测。下面 300 次骨架重采样区间是选型后诊断，不能修正搜索偏差；留一折诊断也不是完整嵌套验证。', '',
              '| 任务 | valid 增量 95% 区间 | 重采样正收益占比 | 留一折选型平均增量 |',
              '| --- | --- | ---: | ---: |']
    for x in uncertainty['tasks']:
        lo, hi = x['post_selection_scaffold_bootstrap_95']
        fmt = (lambda a: f'{a*100:+.3f}%') if x['metric'] == 'MAE' else (lambda a: f'{a:+.6f}')
        lines.append(f"| {x['task']} | [{fmt(lo)}, {fmt(hi)}] | {x['bootstrap_positive_fraction']:.1%} | {fmt(x['held_fold_mean'])} |")
    negatives = [(x['task'], j+1, g) for x in selected for j, g in enumerate(x['selection']['cv_gains']) if g < 0]
    lines += ['', '入选选项中的负收益折：', '']
    lines += [f'- {task} 第 {fold} 折：{gain:+.6f}' for task, fold, gain in negatives] or ['无。']
    lines += ['', '官方总分公式、任务权重和分项得分未知，以上增量不是平台分数预测。即使验证有收益，盲测仍可能回退。最后报告第一名为 52.723518，超过其一分的目标为 53.723518，尚未达到。', '',
              '## 交付核验', '', f'`{file.relative_to(ROOT).as_posix()}`', '',
              f"16,379 行、22 项任务、6 列，{verify['bytes']:,} 字节，UTF-8 BOM 与 CRLF。非预测字段逐行与官方模板相同。远程重载所有入选模型、重新构建特征并验证；目标任务属性块与测试标签污染不改变特征。本机独立重算融合，检查身份、行序、唯一键、数值范围、文件哈希与源代码哈希，通过。", '',
              f"SHA256：`{verify['sha256']}`。", '',
              'v36 作为最后确认得分的回退文件继续保留；本轮只提供一个候选 CSV。']
    report = ROOT / 'admet_diagnostics/v37_report.md'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    path = ROOT / 'admet_current_candidate.json'
    state = json.loads(path.read_text(encoding='utf-8'))
    assert state['platform_anchor_release'] == 'v36' and state['platform_anchor_score'] == 50.470960
    state.update(experimental_release='v37', experimental_submission=str(file), experimental_sha256=verify['sha256'],
                 experimental_status='EXPERIMENTAL_AWAITING_PLATFORM', candidate_platform_score=None,
                 awaiting='User platform evaluation of v37', report='admet_diagnostics/v37_report.md',
                 verification='admet_submissions/v37/local_verification.json',
                 remote_model_verification='admet_model_outputs_v37/model_replay.json',
                 ready_for_platform_evaluation=True, changed_task_count=len(selected),
                 automatically_uploaded=False, local_gpu_training=False, target_achieved=False)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(file))
    print(str(report))


if __name__ == '__main__':
    main()
