"""Finalize verified v379 evidence without changing training or model selection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    exp = ROOT / 'experiments/regression9_v379'
    best = read(ROOT / 'docs/current_best_after_v374_20261105.json')
    delivery = read(ROOT / 'docs/v379_delivery.json')
    verified = read(exp / 'outputs/independent_verification.json')
    completion = read(exp / 'completion_receipt.json')
    curves = read(ROOT / 'docs/v379_learning_curves.json')
    assert sha(ROOT / best['csv']) == best['csv_sha256']
    assert verified['status'] == completion['status'] == 'PASS'
    assert read(exp / 'run_status.json')['stage'] == 'EVIDENCE_READY'
    assert delivery['signature'] == verified['signature'] == completion['signature']
    assert completion['no_training_processes_running']
    report = dict(status='COMPLETE', version='v379',
                  completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  official_best=51.271, official_new_score=None, delivery=delivery,
                  remote_completion_receipt=completion,
                  late_better_on_both_parts=curves['late_better_on_both_parts'],
                  total_task_base_pairs=curves['total_task_base_pairs'],
                  goal_achieved_local=delivery['goal_mean_5percent_achieved_local'],
                  all_fitting_remote=True)
    save(ROOT / 'docs/v379_progress.json', report)
    lines = ['# v379 完整实验与交接', '',
             '官方已知最好仍为51.271。9项连续任务纳入MAE研究，其中只有5项直接以MAE计分。', '',
             '两种配置、两个种子、两个开发分区：36条曲线、72个完整种子训练，每个训练6000步；保存1000/3000/6000。全部模型拟合在指定25870远端RTX3090。',
             '小容量Transformer、参数指数平均，对照中位数L1与有序分位数辅助；4项排序任务联合成对排序损失。全部开发曲线结束才选型，冻结后确认，失败没有重选配方。', '',
             f"独立重载和查询标签扰动核验：{verified['models']}个预测包，最大重放误差{verified['full_replay_max_error']:.4g}。输入、源码、模型哈希已核验。", '',
             f"长预算比短预算同时改善两个开发分区：{curves['late_better_on_both_parts']}/{curves['total_task_base_pairs']}个任务/方法组合。此项为描述分析，不参与确认重选。", '',
             '逐任务结果见`V379_REGRESSION9_RESULTS.md`及`v379_delivery.json`；学习曲线见`v379_learning_curves.png`。', '',
             f"采用任务：{', '.join(delivery['accepted_tasks']) or '无'}。",
             f"9项平均MAE降幅（fold0/fold1/fold2/valid）：{' / '.join(f'{x:.3%}' for x in delivery['nine_task_mean_relative_MAE_gain_by_part'].values())}。",
             f"5项直接计分平均：{' / '.join(f'{x:.3%}' for x in delivery['five_scored_task_mean_relative_MAE_gain_by_part'].values())}。",
             '9项平均5%目标：'+('局部验证达到，官方未知。' if delivery['goal_mean_5percent_achieved_local'] else '未达到。'), '',
             ('实验CSV：`'+delivery['csv']+'`；新官方分数未知。' if delivery['csv'] else '没有满足交付检查的更新，没有生成新CSV；保留现有51.271关联文件。'), '',
             '四项Spearman任务的MAE参考是拟合标签分位数校准后的历史预测秩，不能称为历史模型真实原单位误差。v379用同一原单位融合输出计算MAE、Spearman及提交。历史对照部分为proxy，公开标签反复使用，确认区不等于新盲测。', '',
             '材料整理覆盖现有相关源码及精选原始论文页面，没有穷尽全球研究。97项源码声明不是97个独立成功模型，重复方法和失败结果均保留。', '',
             '条件算术：五项官方MAE同时降5%、其余17项固定，约51.561；不能据此承诺达到52。', '',
             '后续应优先补足完整可重训基线及真正分组OOF，进而比较任务专家残差与低自由度融合；v372已有四专家嵌套融合失败，不能把相同配置换名重复。需要改变可检验的输入、损失或数据覆盖机制，并保持预算及消融。', '']
    (ROOT / 'docs/V379_COMPLETE_RESULTS_AND_HANDOFF.md').write_text('\n'.join(lines), encoding='utf-8')
    state = read(ROOT / 'CURRENT_STATUS.json')
    state.update(latest_completed_model_branch='v379', latest_completed_model_branch_is_proven_best=False,
                 latest_aggregate_plan='v377_v378_v379', aggregate_status='REGRESSION9_V379_COMPLETE',
                 remote_training_running=False, current_experiment_progress='docs/v379_progress.json',
                 official_score=51.271,
                 goal_nine_task_mean_MAE_five_percent_achieved=delivery['goal_mean_5percent_achieved_local'],
                 goal_five_scored_mean_MAE_five_percent_achieved=delivery['five_scored_goal_achieved_local'])
    if delivery['csv']:
        state['latest_MAE_experimental_submission'] = delivery
    save(ROOT / 'CURRENT_STATUS.json', state)
    prefix = '\n'.join(['# 最新补充：v379完整实验结束', '',
                        '官方最好仍为51.271。结果及逐任务证据：`docs/V379_COMPLETE_RESULTS_AND_HANDOFF.md`。',
                        '9项平均MAE降5%：'+('局部验证达到，官方未知。' if delivery['goal_mean_5percent_achieved_local'] else '未达到。'),
                        '训练和核验已退出，空闲仅对应completion_receipt.json的记录时刻。', '', '---', ''])
    old = (ROOT / 'HANDOFF_CURRENT.md').read_text(encoding='utf-8')
    if not old.startswith('# 最新补充：v379完整实验结束'):
        old = prefix + '\n' + old
    for path in (ROOT / 'HANDOFF_CURRENT.md', ROOT / 'docs/HANDOFF_CURRENT.md'):
        path.write_text(old, encoding='utf-8')
    files = [ROOT / 'CURRENT_STATUS.json', ROOT / 'HANDOFF_CURRENT.md', ROOT / 'docs/HANDOFF_CURRENT.md',
             ROOT / 'releases/v377_v378_complete_handoff.tar.gz',
             ROOT / 'releases/v379_distribution_delta.tar.gz', ROOT / 'releases/v379_prelaunch_source_v1.tar.gz',
             ROOT / 'docs/v379_learning_curves.png', ROOT / 'tools/v377_deliver_regression.py',
             ROOT / 'tools/v377_analyze_learning_curves.py', ROOT / 'tools/v378_research_vs_deployment.py']
    files += list((ROOT / 'tools').glob('v379_*.py'))
    files += list((ROOT / 'docs').glob('v379_*.json'))
    files += list((ROOT / 'docs').glob('V379_*.md'))
    files += [p for p in exp.rglob('*') if p.is_file() and
              ('/outputs/' in p.as_posix() or p.suffix in ('.json', '.py', '.log'))]
    if delivery['csv']:
        files += list((ROOT / delivery['csv']).parent.glob('*'))
    files = sorted(set(p for p in files if p.name not in ('v379_release.json', 'v379_release_manifest.json')))
    manifest = ROOT / 'docs/v379_release_manifest.json'
    entries = [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p), bytes=p.stat().st_size) for p in files]
    save(manifest, dict(files=entries, credentials_included=False,
                        inherited_inputs='Verified v377/v378 complete handoff nested in this archive.',
                        nonselected_checkpoints='Retained on authorized remote server.'))
    archive = ROOT / 'releases/v379_complete_handoff.tar.gz'
    with tarfile.open(archive, 'w:gz') as tf:
        for p in files + [manifest]:
            tf.add(p, arcname=p.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for row in entries:
            assert hashlib.sha256(tf.extractfile(row['path']).read()).hexdigest() == row['sha256']
    release = dict(status='PASS', archive=archive.relative_to(ROOT).as_posix(), sha256=sha(archive),
                   bytes=archive.stat().st_size, members=len(entries)+1, all_member_hashes_verified=True)
    save(ROOT / 'docs/v379_release.json', release)
    print(json.dumps(dict(status='COMPLETE', csv=delivery['csv'], goal=report['goal_achieved_local'], release=release)))


if __name__ == '__main__':
    main()
