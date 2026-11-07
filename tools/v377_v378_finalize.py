"""Summarize two verified experiments, preserving the official best and history."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    best = read(ROOT / 'docs/current_best_after_v374_20261105.json')
    assert sha(ROOT / best['csv']) == best['csv_sha256']
    reports = []
    for version in (377, 378):
        exp = ROOT / f'experiments/regression9_v{version}'
        delivery = read(ROOT / f'docs/v{version}_delivery.json')
        verified = read(exp / 'outputs/independent_verification.json')
        completion = read(exp / 'completion_receipt.json')
        assert verified['status'] == completion['status'] == 'PASS'
        assert verified['signature'] == delivery['signature']
        assert completion['no_training_processes_running']
        assert read(exp / 'run_status.json')['stage'] == 'EVIDENCE_READY'
        curves = read(ROOT / f'docs/v{version}_learning_curves.json')
        report = dict(status='COMPLETE', version=f'v{version}',
                      completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      training_host='connect.nmb2.seetacloud.com:25870',
                      official_best=best['official_user_reported'],
                      official_new_score=None, delivery=delivery,
                      models_verified=verified['models'],
                      remote_idle=completion,
                      late_better_on_both_parts=curves['late_better_on_both_parts'],
                      total_task_base_pairs=curves['total_task_base_pairs'])
        save(ROOT / f'docs/v{version}_progress.json', report)
        reports.append(report)
    combined = dict(status='COMPLETE', official_best=best,
                    experiments=reports,
                    nine_task_goal_achieved_local=any(r['delivery']['goal_mean_5percent_achieved_local'] for r in reports),
                    five_scored_task_goal_achieved_local=any(r['delivery']['five_scored_goal_achieved_local'] for r in reports),
                    no_new_official_evaluation=True, all_fitting_remote=True,
                    independent_experiments_not_blindly_combined=True)
    next_path=ROOT/'docs/v379_progress.json'
    next_progress=read(next_path) if next_path.exists() else {}
    next_running=next_progress.get('status') in ('DEVELOPMENT','CONFIRMATION','VERIFYING','STARTED')
    save(ROOT / 'docs/v377_v378_summary.json', combined)
    lines = ['# 九项连续任务：v377 / v378 完整实验结果', '',
             f"官方已知最好：{best['official_user_reported']}。本轮没有自动提交，实验文件的新官方分数未知。", '',
             '实际范围为5项直接MAE计分、4项Spearman连续任务。平均按9项相对降幅等权计算，拒绝任务计零；另报5项直接计分平均。', '',
             'v377：原尺度中位数树、两种深度、加权asinh，双种子各完整3840轮，保存640/1920/3840。',
             'v378：六指纹视图和数值token Transformer，比较直接预测与32邻居标签检索，双种子各完整4800步，保存800/2400/4800。',
             '两轮都在全部开发曲线结束后选型；fold0/1选定唯一配置与权重，再检查fold2/valid，失败未换配方。', '',
             '|实验|采用任务|9项平均相对MAE下降（fold0/fold1/fold2/valid）|5项直接计分平均|9项平均5%目标|',
             '|---|---|---|---|---|']
    for report in reports:
        delivery = report['delivery']
        gains = '/'.join(f'{x:.3%}' for x in delivery['nine_task_mean_relative_MAE_gain_by_part'].values())
        five = '/'.join(f'{x:.3%}' for x in delivery['five_scored_task_mean_relative_MAE_gain_by_part'].values())
        lines.append('|'+ '|'.join([report['version'], ', '.join(delivery['accepted_tasks']) or '无', gains, five,
                                    '局部验证达到，官方未知' if delivery['goal_mean_5percent_achieved_local'] else '未达到'])+'|')
    lines += ['', '四项排序任务的历史输出含任意尺度。本轮MAE对照使用拟合标签经验分位数校准的秩参考，不能称为旧模型真实原单位误差；采用更新还要求四分区Spearman改善。',
              '历史控制仍部分为proxy；公开标签已反复使用，确认区间并非新盲测。不能把本轮平均数当作隐藏测试降幅。', '',
              '逐任务完整原始MAE、Spearman、成组置信区间、来源和哈希见`v377_delivery.json`、`v378_delivery.json`及各实验输出。',
              '研究依据、已尝试路线和预算原则见`V376_REGRESSION_RESEARCH_AND_LONG_HORIZON_PLAN.md`。', '',
              '按最新卡仅做条件算术：五项MAE都下降5%、其余17项不变，总分约51.561；同幅下降约12.1085%才对应52。这不是模型效果预测。', '',
              '训练预算诊断：']
    for report in reports:
        lines += [f"- {report['version']}：{report['late_better_on_both_parts']}/{report['total_task_base_pairs']}个任务/方法组合，在固定融合权重下最长训练档比最短档同时改善两个开发分区。该描述不参与确认重选。"]
    lines += ['', '可提交文件：']
    candidates = [r['delivery'] for r in reports if r['delivery']['csv']]
    if candidates:
        for item in candidates:
            lines += [f"- {item['version']}：`{item['csv']}`，标记为实验文件，新官方分数未知。"]
        lines += ['两轮均独立对照v374构建，不能未经验证直接叠加。']
    else:
        lines += ['本轮没有满足四分区方向一致的更新，没有生成重复基线冒称新CSV。继续保留v374关联51.271文件。']
    lines += ['', '后续路线依据完整学习曲线再定：持续改善的任务可预注册更长预算；训练误差继续下降而验证恶化的任务优先正则化和表示。正式训练可复用的完整历史集成OOF仍是关键缺口，不能用重代入预测训练堆叠。', '']
    if next_running:
        lines += ['后续v379已启动，尚无结论；单层32维Transformer、中位数/分位数目标、参数平均与排序联合损失，见`V379_REGULARIZED_DISTRIBUTION_EXPERIMENT.md`。本报告完成回执证明的是v377/v378结束时的空闲状态，不代表v379启动后服务器空闲。', '']
    document = '\n'.join(lines)
    (ROOT / 'docs/V377_V378_REGRESSION9_RESULTS_AND_HANDOFF.md').write_text(document, encoding='utf-8')
    state_path = ROOT / 'CURRENT_STATUS.json'
    state = read(state_path)
    state.update(latest_completed_model_branch='v378', latest_completed_model_branch_is_proven_best=False,
                 latest_aggregate_plan='v377_v378', aggregate_status='COMPLETE_REGRESSION9_EXPERIMENTS',
                 remote_training_running=next_running, official_score=best['official_user_reported'],
                 current_experiment_progress='docs/v377_v378_summary.json',
                 latest_regression9_round=combined,
                 goal_nine_task_mean_MAE_five_percent_achieved=combined['nine_task_goal_achieved_local'],
                 goal_five_scored_mean_MAE_five_percent_achieved=combined['five_scored_task_goal_achieved_local'])
    if candidates:
        state['latest_MAE_experimental_submission'] = candidates[-1]
    if next_running:
        state.update(latest_aggregate_plan='v377_v378_v379',aggregate_status='REGRESSION9_V379_TRAINING',
                     current_experiment_progress='docs/v379_progress.json')
    save(state_path, state)
    prefix = '\n'.join(['# 最新补充：九项连续任务完整实验', '',
                        f"官方最好仍为{best['official_user_reported']}；v377/v378已结束、独立重放通过。"+(' 后续v379正在训练。' if next_running else ' 远端在完成回执时间已空闲。'), '',
                        '详细交付：`docs/V377_V378_REGRESSION9_RESULTS_AND_HANDOFF.md`。',
                        '9项平均相对MAE降低5%：'+('局部验证达到，官方未知。' if combined['nine_task_goal_achieved_local'] else '未达到。'), '', '---', ''])
    path = ROOT / 'HANDOFF_CURRENT.md'
    old = path.read_text(encoding='utf-8')
    if not old.startswith('# 最新补充：九项连续任务完整实验'):
        old = prefix + '\n' + old
    for path in (ROOT/'HANDOFF_CURRENT.md', ROOT/'docs/HANDOFF_CURRENT.md'):
        path.write_text(old, encoding='utf-8')
    # Exact full input archive plus current sources/results/final checkpoints;
    # large arrays deduplicated by inclusion of the earlier verified handoff.
    files = [ROOT/'CURRENT_STATUS.json', ROOT/'HANDOFF_CURRENT.md', ROOT/'docs/HANDOFF_CURRENT.md',
             ROOT/'tools/verify_submission.py', ROOT/'tools/repack_upload_template.py',
             ROOT/'docs/platform_scorecard_v374_20261105.json', ROOT/'docs/current_best_after_v374_20261105.json',
             ROOT/'releases/v374_v375_complete_handoff.tar.gz',
             ROOT/'releases/v377_regression9_delta.tar.gz', ROOT/'releases/v378_neural_delta.tar.gz',
             ROOT/'docs/v377_v378_learning_curves.png']
    for version in (376, 377, 378):
        files += list((ROOT/'tools').glob(f'v{version}_*.py'))
        files += list((ROOT/'docs').glob(f'v{version}_*.json'))
        files += list((ROOT/'docs').glob(f'V{version}_*.md'))
    for version in (377, 378):
        exp = ROOT/f'experiments/regression9_v{version}'
        files += [p for p in exp.rglob('*') if p.is_file() and (
            '/outputs/' in p.as_posix() or p.suffix in ('.json', '.py', '.log'))]
    for delivery in candidates:
        files += list((ROOT/delivery['csv']).parent.glob('*'))
    files = sorted(set(files))
    files = [p for p in files if p.name not in ('v377_v378_release.json', 'v377_v378_release_manifest.json')]
    entries = [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p), bytes=p.stat().st_size) for p in files]
    manifest = ROOT/'docs/v377_v378_release_manifest.json'
    save(manifest, dict(files=entries, inputs='Current delta archive, plus earlier complete verified handoff for inherited arrays.',
                        credentials_included=False, nonselected_checkpoints_remain_remote=True))
    archive = ROOT/'releases/v377_v378_complete_handoff.tar.gz'
    with tarfile.open(archive, 'w:gz') as tf:
        for p in files+[manifest]:
            tf.add(p, arcname=p.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for row in entries:
            assert hashlib.sha256(tf.extractfile(row['path']).read()).hexdigest() == row['sha256']
    release = dict(status='PASS', archive=archive.relative_to(ROOT).as_posix(), sha256=sha(archive),
                   bytes=archive.stat().st_size, members=len(files)+1, all_member_hashes_verified=True)
    save(ROOT/'docs/v377_v378_release.json', release)
    print(json.dumps(dict(status='COMPLETE', official_best=best['official_user_reported'],
                          goal=combined['nine_task_goal_achieved_local'],
                          csvs=[d['csv'] for d in candidates], archive=release)))


if __name__ == '__main__':
    main()
