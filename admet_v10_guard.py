"""Apply cumulative regression guard to the delivered v10; immutable input files.

This is a safety correction, not a newly trained model or evidence of score gain.
Any task that fails an observed cumulative gate returns wholly to scored v7.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path

NAME='我从山北来带着兰花草-小分子药物预测技术创新赛-20260925.csv'
EXPECTED={7:'88e81b9d16e48cb941f18c5fd6c4a49e067c73659de808e896e51dea8de5acf0',
          10:'d9d7a9e725d2083e799229e1c988ee396d598453d58c331c2143d77219ab9a4d'}


def sha(blob):return hashlib.sha256(blob).hexdigest()


def json_write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def rows(blob):return list(csv.reader(io.StringIO(blob.decode('utf-8-sig'),newline='')))


def main():
    source={v:Path(f'admet_submissions/v{v}')/NAME for v in EXPECTED}
    blobs={v:p.read_bytes() for v,p in source.items()}
    for v,b in blobs.items():assert sha(b)==EXPECTED[v]
    audit_file=Path('admet_diagnostics/v10_anchor_audit.json');audit=json.loads(audit_file.read_text(encoding='utf-8'))
    assert audit['anchor_sha256']==EXPECTED[7] and audit['final_sha256']==EXPECTED[10]
    fallback={r['task'] for r in audit['violations']}
    if audit['insufficient_coverage']:raise ValueError('Coverage gaps require explicit assessment, not a pass')
    anchor=rows(blobs[7]);old=rows(blobs[10]);new=[r.copy() for r in old]
    changed_rows=0
    for a,b,c in zip(anchor[1:],old[1:],new[1:]):
        assert a[:4]+a[5:]==b[:4]+b[5:]
        if c[0] in fallback:
            c[4]=a[4];changed_rows+=int(c[4]!=b[4])
    stream=io.StringIO(newline='');csv.writer(stream,lineterminator='\r\n').writerows(new)
    blob=b'\xef\xbb\xbf'+stream.getvalue().encode('utf-8')
    dest=Path('admet_submissions/v10_guard')/NAME;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.read_bytes()!=blob:raise RuntimeError('Refuse overwrite of different existing guarded CSV')
    dest.write_bytes(blob)
    # Independent string/template readback, no pandas numeric reformatting.
    actual=rows(dest.read_bytes());template=rows(Path('admet_official/submission_template.csv').read_bytes())
    assert len(actual)==len(anchor)==len(old)==len(template)==16380
    assert actual[0]==template[0]==anchor[0]==old[0]
    assert blob.count(b'\r\n')==16380 and blob.count(b'task,sample_idx,drug_id,smiles,y_pred,metric')==1
    changed=set();all_tasks=set()
    for a,b,c,t in zip(anchor[1:],old[1:],actual[1:],template[1:]):
        assert len(c)==6 and c[:4]+c[5:]==t[:4]+t[5:]
        assert c==(a if c[0] in fallback else b)
        value=float(c[4]);assert math.isfinite(value)
        if c[5] in ['AUROC','AUPRC']:assert 0<=value<=1
        all_tasks.add(c[0])
        if c!=a:changed.add(c[0])
    assert len(all_tasks)==22 and changed==set(audit['changed_tasks'])-fallback
    # Read recorded full metrics and check thresholds again against the exact retained routes.
    with Path('admet_diagnostics/v10_anchor_audit.csv').open(encoding='utf-8-sig',newline='') as f:
        checks=list(csv.DictReader(f))
    for r in checks:
        assert r['coverage_sufficient']=='True'
        if r['task'] in fallback:
            for k in list(r):
                if k.startswith('after_'):r[k]=r['before_'+k[len('after_'):]]
            r.update(primary_gain='0',secondary_gain='0',primary_pass='True',secondary_pass='True')
        assert float(r['primary_gain'])>=-float(r['primary_tolerance'])-1e-12
        tol=.03 if r['secondary_metric']=='mae' else .015
        assert float(r['secondary_gain'])>=-tol-1e-12
    with Path('admet_diagnostics/v10_guard_checks.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
    primary_loss=[]
    for task in sorted(fallback):
        r=next(x for x in checks if x['task']==task and x['split']=='valid' and x['distant']=='False')
        with Path('admet_diagnostics/v10_vs_scored_v7.csv').open(encoding='utf-8-sig',newline='') as f:
            old_summary=next(x for x in csv.DictReader(f) if x['task']==task)
        primary_loss.append(dict(task=task,metric=r['metric'],v10_valid=float(old_summary['v10_valid']),guarded_valid=float(old_summary['v7_valid'])))
    record=dict(status='VERIFIED_GUARDED_RELEASE',release='v10_guard',submission=str(dest.resolve()),sha256=sha(blob),
        bytes=len(blob),rows=16379,columns=6,tasks=22,encoding='UTF-8 BOM',line_endings='CRLF',
        score_anchor_version=7,score_anchor=49.343821,target_score=51.584929,platform_score=None,target_achieved=False,
        source_v10_sha256=EXPECTED[10],anchor_sha256=EXPECTED[7],source_audit_sha256=sha(audit_file.read_bytes()),
        fallback_tasks=sorted(fallback),prediction_rows_reverted=changed_rows,retained_improved_tasks=sorted(changed),
        unchanged_vs_v7=sorted(all_tasks-changed),observed_cumulative_checks=len(checks),failures_after_guard=0,
        validation_primary_tradeoff=primary_loss,automatic_submission=False,local_gpu_training=False,
        provenance='Retained tasks use independently replayed v10; fallback task copies scored v7 strings exactly.',
        limitation='Post-selection regression correction, not an untouched evaluation or predicted platform score gain.')
    json_write('admet_release_v10_guard/verification.json',record)
    for v,p in source.items():assert sha(p.read_bytes())==EXPECTED[v]
    lines=['# v10_guard：统一累计回退门槛后的提交候选','',
        '这是一项回退保护修正，没有新训练模型，也不代表平台分数一定上升。v7 和原始 v10 均保持原文件。', '',
        '## 为什么修正','',
        '逐轮比较会遗漏两类问题：小幅回退累积，以及旧轮次采用的方案没有经过新轮次新增的检查。本次将最终 v10 与平台已评分的 v7 直接比较，覆盖 7 个改动任务的三折、官方 valid、额外切分及对应低相似度组，共 70 项比较。', '',
        'PPBR 在 v8 和 v10 被连续修改，累计检查仍通过。唯一不满足新增统一门槛的是 v9 引入的 DILI：fold_1 的 42 个低相似度分子，辅助 AUPRC 由 0.875628 降至 0.855106，下降 0.020522，超过 0.015 容差。此前报告已披露这项回退，但没有将 v10 的新增辅助门槛追溯应用到旧方案。', '',
        '修正版把 DILI 全部 96 条测试预测恢复为已评分 v7 的原字符串，其余预测保留 v10。未根据标签逐个挑选分子，未重试混合权重。', '',
        '## 必须明确的取舍','',
        'DILI 的整体 valid AUROC 因恢复旧模型从 0.914286 回到 0.903297；其他任务验证结果不变。修正优先满足避免局部明显回退的要求，不能称为所有指标均优于原始 v10，也不能保证平台分数更高。', '',
        '## 核验结果','',
        f'- 最终保留 {len(changed)} 个任务改进，{len(all_tasks-changed)} 个任务的预测字符串与 v7 一致。',
        '- 70 项累计主/辅助指标检查均满足既定容差，未出现覆盖不足。通过容差不代表每一指标都提高。',
        '- 保留的预测继承 v10 独立模型重载证据；恢复的预测逐字等于已成功提交的 v7。CSV 元数据、顺序、编码、唯一表头与官方模板核对通过。',
        f'- 文件：{dest.resolve()}',f'- SHA256：{sha(blob)}','',
        '验证数据反复使用，存在选型偏差。最后平台实测仍为 49.343821，目标仍为 51.584929；只有该文件的平台实测才能确认效果。继续使用成功的整文件上传方法，没有自动上传。','']
    Path('admet_diagnostics/v10_guard_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
