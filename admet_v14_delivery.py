"""Local evidence reconciliation and final delivery; no model fitting or upload."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('admet_release_v14')
    final = read(root/'local_verification.json')
    assert final['status'] == 'VERIFIED_LOCAL_AND_REMOTE'
    upload = Path(final['local_submission'])
    assert sha(upload) == final['sha256']
    routes = read(root/'routing.json')
    sources = {v: read(f'admet_release_v{v}/independent_verification.json') for v in [12, 13]}
    audits = {v: read(f'admet_diagnostics/v{v}_anchor_audit.json') for v in [12, 13]}
    for v in [12, 13]:
        assert read(f'admet_release_v{v}/local_verification.json')['status'] == 'VERIFIED_LOCAL_AND_REMOTE'
        assert sources[v]['status'] == 'VERIFIED' and sources[v]['independent_replay']
        assert audits[v]['status'] == 'PASSED'
        assert not audits[v]['violations'] and not audits[v]['insufficient_coverage']
        assert sha(sources[v]['submission']) == sources[v]['sha256']
    rows = table(upload)
    by_source = {v: table(sources[v]['submission']) for v in [12, 13]}
    baseline = next(Path('admet_submissions/v11').glob('*.csv'))
    assert sha(baseline) == '082183f0c462081ea8735a41ec2e47326ef4cdcaf9d1188ae86a006f8057122d'
    by_source[11] = table(baseline)
    routing = {r['task']: r for r in routes}
    assert len(routing) == 22 and len(rows) == 16379
    for i, row in enumerate(rows):
        # Includes exact prediction strings, not only approximate numeric equality.
        assert row == by_source[routing[row['task']]['version']][i]
    checks = table('admet_diagnostics/v14_anchor_checks.csv') if final['changed'] else []
    assert len(checks) == final['anchor_checks'] == 10*len(final['changed'])
    for check in checks:
        assert all(check[k] == 'True' for k in ['coverage_sufficient', 'primary_pass', 'secondary_pass'])
    summary = []
    trials = []
    for v in [12, 13]:
        decisions = [read(p) for p in sorted(Path(f'admet_model_outputs_v{v}').glob('*/decision.json'))]
        assert len(decisions) == 22
        assert len(list(Path(f'admet_model_outputs_v{v}').glob('*/*_cv.json'))) == 66
        assert read(f'admet_model_outputs_v{v}/finish_status.json')['status'] == 'COMPLETE_VERIFIED'
        trials.append(dict(version=v, tasks=22, model_families=3, outer_fits=198,
            accepted=sources[v]['accepted'],
            rejections=dict(Counter(d['reason'] for d in decisions if not d['accepted']))))
        for row in table(f'admet_diagnostics/v{v}_comparison.csv'):
            if routing[row['task']]['version'] == v:
                row['source_version'] = v
                summary.append(row)
    assert {r['task'] for r in summary} == set(final['changed'])
    caveat = ('OOF, official validation and repeat split were reused across many rounds. '
              'Observed gains are selection-biased and cannot establish a platform-score gain '
              'or guarantee no hidden-set regression.')
    run = dict(status='experiments_complete_candidate_verified_awaiting_platform_score',
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        baseline_release='v11', baseline_score=49.381036, required_gain=.6,
        minimum_requested_score=49.981036, ultimate_target=51.584929,
        candidate_version='v14', candidate_score=None, target_achieved=False,
        remote_directory='/root/autodl-tmp/admet_20260925', remote_training_only=True,
        local_training=False, automatic_submission=False, score_guarantee=False,
        trial_summary=trials, adopted=final['changed'], retained=final['unchanged'],
        anchor_checks=len(checks), tolerance_failures=0, insufficient_coverage=0,
        submission=str(upload.resolve()), sha256=final['sha256'], bytes=upload.stat().st_size,
        limitation=caveat)
    save('admet_v14_remote_run.json', run)
    save('admet_iteration_20260925.json', run)
    save('admet_current_candidate.json', dict(recommended_release='v14',
        submission=str(upload.resolve()), sha256=final['sha256'],
        platform_anchor_release='v11', platform_anchor_score=49.381036,
        candidate_platform_score=None, next_version_minimum_target=49.981036,
        requested_gain=.6, target_score=51.584929, target_achieved=False,
        awaiting='User-reported official score for this exact verified v14 file',
        report='admet_diagnostics/v14_delivery_report.md',
        verification='admet_release_v14/local_verification.json', limitation=caveat))
    lines = ['# 本轮远程模型迭代交付', '',
        '已评分基线 v11：49.381036。本轮要求至少 +0.6，即 49.981036；最终目标 51.584929。新文件未提交，尚无官方分数，不能宣称达标。', '',
        '两条独立路线均直接对比已评分 v11：内部骨架调参的梯度提升/子结构核，以及冻结 MoLFormer 表示的核/残差神经网络。共完成 22 个任务 × 6 个模型家族 × 3 折 = 396 次外层拟合；内部选择和发布复拟合另计。所有训练和表示推理在远程服务器完成。', '',
        f'最终采用 {len(summary)} 项任务改动；其余 {22-len(summary)} 项保留 v11 原始预测字符串。v14 仅组合已核验的结果，没有再次寻找混合权重。', '',
        '|任务|主指标|v11 验证集|新验证集|来源/模型|新模型权重|',
        '|---|---|---:|---:|---|---:|']
    for r in summary:
        key=r['metric'].lower()
        lines.append(f"|{r['task']}|{r['metric']}|{float(r['before_'+key]):.6f}|{float(r['after_'+key]):.6f}|v{r['source_version']} / {r['family']}|{r['alpha']}|")
    lines += ['',
        f'相对已评分 v11 复核 {len(checks)} 项整体/低相似度组比较，超出既定容差 0 项，覆盖不足 0 项。独立进程重载发布模型并复算测试预测；本地再核验来源、文件哈希和逐行模板。', '',
        '“通过门槛”不等于所有指标均增加。官方验证集主指标不能下降；各折和额外切分容许 MAE 最多相对下降 1% 或排名指标下降 0.005。低相似度组主指标容差为 MAE 3% 或排名指标 0.01；辅助指标另受 MAE 3% 或排名指标 0.015 的约束。', '',
        '|发生的小幅退步|切分|低相似度组|指标|变化|', '|---|---|---|---|---:|']
    regressions = []
    for c in checks:
        for label in ['primary', 'secondary']:
            gain = float(c[label+'_gain'])
            if gain < 0:
                metric=c['metric'] if label=='primary' else c['secondary_metric']
                regressions.append((c, metric, gain))
                lines.append(f"|{c['task']}|{c['split']}|{c['distant']}|{metric}|{gain:.6f}|")
    if not regressions:lines.append('|无|—|—|—|—|')
    lines += ['', 'MAE 的变化为相对改善比例；排名指标为绝对差值，负值代表下降。详细数值见 v14_anchor_checks.csv。', '',
        '验证划分被多轮反复用于选型，有选择偏差。条件 bootstrap 仅反映已选方案在复用验证集上的波动，不是平台分数区间；不能将上表换算为 +0.6 分，也不能保证隐藏集不退步。', '',
        '目前单靠继续叠加相似模型，缺少达到下一分数门槛的证据。本轮扩大模型结构后仍有大量候选在官方验证或低相似度组退步而被拒绝。没有降低门槛来制造提升，也没有使用外部 ADMET 实验标签或盲测标签。', '',
        '提交格式：16,379 条数据、22 项任务、6 列，UTF-8 BOM、CRLF、唯一表头，所有非预测字段与官方模板一致。继续使用之前成功的整文件上传方式；刷新网页会恢复网站分片设置。没有自动上传。', '',
        f'唯一推荐提交文件：{upload.resolve()}', f'SHA256：{final["sha256"]}', '']
    Path('admet_diagnostics/v14_delivery_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(run, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
