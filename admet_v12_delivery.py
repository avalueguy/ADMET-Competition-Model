"""Record verified v12 evidence locally; no training and no platform upload."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    root=Path('admet_release_v12');local=read_json(root/'local_verification.json')
    audit=read_json('admet_diagnostics/v12_anchor_audit.json')
    assert local['status']=='VERIFIED_LOCAL_AND_REMOTE'
    assert audit['status']=='PASSED' and not audit['violations']
    upload=Path(local['submission']);assert hashlib.sha256(upload.read_bytes()).hexdigest()==local['sha256']==audit['final_sha256']
    routes=read_json(root/'routing.json');accepted=[r for r in routes if r['accepted']]
    assert {r['task'] for r in accepted}==set(local['changed'])
    with Path('admet_diagnostics/v12_comparison.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==22
    for row in rows:
        key=row['metric'].lower();before=float(row['before_'+key]);after=float(row['after_'+key])
        assert before>=after if key=='mae' else after>=before
    run=dict(status='complete_verified_not_submitted',version=12,generated_at_utc=datetime.now(timezone.utc).isoformat(),
        remote_directory='/root/autodl-tmp/admet_20260925',baseline_platform_score=49.381036,target_score=51.584929,next_version_minimum_target=49.981036,requested_gain=.6,
        platform_score=None,target_achieved=False,remote_training_only=True,automatic_submission=False,
        tasks=22,candidates=66,fold_fits=198,accepted=local['changed'],unchanged=local['unchanged'],
        cumulative_checks=audit['checks'],coverage_gaps=audit['insufficient_coverage'],
        rejection_reasons=dict(Counter(r['reason'] for r in routes if not r['accepted'])),
        submission=str(upload.resolve()),sha256=local['sha256'],bytes=upload.stat().st_size,
        baseline_file_linkage='inferred from user feedback sequence; not independently confirmed by platform hash',
        limitation='Validation reused across rounds; gains cannot be converted to platform points and hidden-set regression remains possible.')
    write_json('admet_v12_remote_run.json',run)
    if accepted:
        write_json('admet_current_candidate.json',dict(recommended_release='v12',submission=str(upload.resolve()),sha256=local['sha256'],
            platform_anchor_release='v11',platform_anchor_score=49.381036,target_score=51.584929,next_version_minimum_target=49.981036,requested_gain=.6,candidate_platform_score=None,
            target_achieved=False,awaiting='User-reported platform score for verified v12 file',
            recommendation_reason='Only task changes passing fixed observed primary, secondary, and distant-subgroup checks are included.',
            report='admet_diagnostics/v12_result_report.md',verification='admet_release_v12/local_verification.json',
            anchor_audit='admet_diagnostics/v12_anchor_audit.json'))
    else:
        run['recommendation']='Keep scored v11. No candidate promoted; duplicate upload offers no model change.'
        write_json('admet_v12_remote_run.json',run)
    lines=['# v12 交付核验','',
        '平台基线：49.381036。本轮目标：至少 49.981036（+0.6）；最终目标：51.584929。v12 尚无平台分数。', '',
        f'远程完成 66 个任务/方案候选、198 次骨架折拟合，最终采用 {len(accepted)} 项变化，保留其余 {22-len(accepted)} 项原始预测字符串。', '',
        '|任务|指标|已评分基线 valid|v12 valid|','|---|---|---:|---:|']
    for row in rows:
        if row['task'] in local['changed']:
            key=row['metric'].lower();lines.append(f"|{row['task']}|{row['metric']}|{float(row['before_'+key]):.6f}|{float(row['after_'+key]):.6f}|")
    lines+=['',f'相对已评分基线的累计复核包含 {audit["checks"]} 项整体/低相似度组比较；超出容差 0 项，覆盖不足 {len(audit["insufficient_coverage"])} 项。', '',
        '容差通过不代表每一指标都提高。历史划分被反复使用，存在选择偏差；条件 bootstrap 见 v12_uncertainty.csv，不能当作平台分数区间。', '',
        f'提交文件：{upload.resolve()}',f'SHA256：{local["sha256"]}', '',
        '远程独立重载模型复算预测，本地再次核对模板、任务顺序、分类取值范围以及未采用任务的原始字符串。',
        '文件包含 16,379 条数据、22 个任务、6 列，UTF-8 BOM、CRLF，表头只出现一次。继续使用已成功的整文件上传方式；没有自动上传。','']
    Path('admet_diagnostics/v12_delivery_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(run,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
