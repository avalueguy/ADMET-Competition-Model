"""Publish only gated improvements and replay saved models independently."""
import argparse
import csv
import io
import json
import shutil
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from admet_v6_runner import remote_only, save, sha, emit, metrics, kind, deltas, secondary_ok, get_metric
from admet_v13_runner import OUT, protocol, baseline, domain_check
from admet_v13_models import load_task, fit, predict

ROOT = Path('admet_release_v13')
PRIOR = Path('admet_release_v11')
OLD_UPLOAD = Path('admet_submissions/v11/我从山北来带着兰花草-小分子药物预测技术创新赛-20260925.csv')
UPLOAD = Path('admet_submissions/v13/我从山北来带着兰花草-小分子药物预测技术创新赛-20260925.csv')


def export_exact_csv(routes):
    """Preserve all original metadata and rejected-task prediction strings byte-for-byte."""
    source = OLD_UPLOAD.read_bytes()
    assert sha(OLD_UPLOAD) == json.loads((PRIOR/'independent_verification.json').read_text())['sha256']
    rows = list(csv.reader(io.StringIO(source.decode('utf-8-sig'), newline='')))
    original = [r.copy() for r in rows]
    assert rows[0] == ['task','sample_idx','drug_id','smiles','y_pred','metric']
    promoted = {r['task'] for r in routes if r['accepted']}
    preds = {t:pd.read_csv(ROOT/t/f'submission_{t}.csv').Y.to_numpy(float) for t in promoted}
    for row in rows[1:]:
        if row[0] in promoted:
            row[4] = repr(float(preds[row[0]][int(row[1])]))
    text = io.StringIO(newline='')
    csv.writer(text, lineterminator='\r\n').writerows(rows)
    blob = b'\xef\xbb\xbf' + text.getvalue().encode('utf-8')
    assert len(rows) == 16380 and blob.count(b'\r\n') == 16380
    for a, b in zip(original, rows):
        assert a[:4] == b[:4] and a[5:] == b[5:]
        if a[0] not in promoted:
            assert a == b
    UPLOAD.parent.mkdir(parents=True, exist_ok=True)
    if UPLOAD.exists() and UPLOAD.read_bytes() != blob:
        raise RuntimeError('Refuse to replace different existing v13 upload file')
    UPLOAD.write_bytes(blob)


def publish():
    p = protocol(); ROOT.mkdir(exist_ok=True); routes = []
    for prior in sorted(x for x in PRIOR.iterdir() if x.is_dir()):
        task = prior.name; dest = ROOT/task; dest.mkdir(exist_ok=True)
        decision_path = OUT/task/'decision.json'
        if not decision_path.exists() and task not in p['tasks']:
            metric = get_metric(task)
            save(decision_path, dict(task=task, accepted=False, reason='outside fixed v13 sparse-kernel pool; retain v11', selection=dict(task=task, family='baseline', alpha=0, metric=metric, kind=kind(metric))))
        decision = json.loads(decision_path.read_text())
        accepted = decision['accepted']
        if accepted:
            d = load_task(task); n = len(d['frames']['train'])+len(d['frames']['valid'])
            y = np.r_[d['frames']['train'].Y, d['frames']['valid'].Y]
            s = decision['selection']; modelpath = dest/'model.joblib'
            previous = json.loads((dest/'selection.json').read_text()) if (dest/'selection.json').exists() else {}
            if modelpath.exists() and previous.get('model_sha256') == sha(modelpath) and previous.get('selection') == s:
                model = joblib.load(modelpath)
            else:
                model = fit(s['family'], d, np.arange(n), y, s['metric'], p)
                joblib.dump(model, modelpath, compress=3)
            pred = predict(model, d, np.arange(n, len(d['smiles'])))
            old = pd.read_csv(prior/f'submission_{task}.csv', dtype={'Drug_ID':str}, keep_default_na=False)
            q = (1-s['alpha'])*old.Y.to_numpy(float)+s['alpha']*pred
            assert np.isfinite(q).all()
            if s['kind'] == 'classification':
                assert ((q>=0)&(q<=1)).all()
            pd.DataFrame(dict(Drug_ID=old.Drug_ID,Y=q)).to_csv(dest/f'submission_{task}.csv', index=False)
            pd.DataFrame(dict(Drug_ID=old.Drug_ID,Y=pred)).to_csv(dest/'candidate_test.csv', index=False)
            v = pd.read_csv(OUT/task/'validation_comparison.csv')
            pd.DataFrame(dict(Y_true=v.Y_true,Y_pred=v.blend)).to_csv(dest/'validation_predictions.csv', index=False)
            if s['family'] in protocol()['families']+['blend_all']:
                save(dest/'source_exclusion_audit.json', model.get('source_audit',[]))
            decision.update(model_sha256=sha(modelpath), feature_key=d['v13_feature_key'])
        else:
            for file in [f'submission_{task}.csv', 'validation_predictions.csv']:
                shutil.copy2(prior/file, dest/file)
        decision.update(base_sha256=sha(prior/f'submission_{task}.csv'), prediction_sha256=sha(dest/f'submission_{task}.csv'))
        save(dest/'selection.json', decision); routes.append(decision)
        emit(stage='v13_publish', task=task, accepted=accepted)
    save(ROOT/'routing.json', routes)
    accepted = [r['task'] for r in routes if r['accepted']]
    export_exact_csv(routes)
    save(ROOT/'verification.json', dict(status='EXPORTED_AWAITING_REPLAY', accepted=accepted, submission=str(UPLOAD),
         sha256=sha(UPLOAD), baseline_score=49.381036, platform_anchor_score=49.381036, platform_score=None, rows=16379,
         protocol_sha256=sha('admet_v13_protocol.json')))
    save(ROOT/'source_manifest.json', [dict(path=str(f), sha256=sha(f)) for f in sorted(Path('.').glob('admet_v13*')) if f.is_file()])


def verify():
    v = json.loads((ROOT/'verification.json').read_text())
    if v['status'] == 'NO_PROMOTION':
        save(ROOT/'independent_verification.json', v); return
    blob = UPLOAD.read_bytes(); assert sha(UPLOAD) == v['sha256']
    assert blob.startswith(b'\xef\xbb\xbf') and blob.count(b'\r\n') == 16380
    assert blob.count(b'task,sample_idx,drug_id,smiles,y_pred,metric') == 1
    new = pd.read_csv(UPLOAD, dtype=str, keep_default_na=False)
    old_upload = pd.read_csv(OLD_UPLOAD, dtype=str, keep_default_na=False)
    template = pd.read_csv('admet_official/submission_template.csv', dtype=str, keep_default_na=False)
    assert new.drop(columns='y_pred').equals(template.drop(columns='y_pred'))
    assert len(new) == 16379 and new.task.nunique() == 22
    checks = []
    for folder in sorted(f for f in ROOT.iterdir() if f.is_dir()):
        task = folder.name; r = json.loads((folder/'selection.json').read_text())
        oldfile = PRIOR/task/f'submission_{task}.csv'; file = folder/f'submission_{task}.csv'
        assert sha(oldfile) == r['base_sha256']; assert sha(file) == r['prediction_sha256']
        result = pd.read_csv(file)
        if r['accepted']:
            s = r['selection']; metric = s['metric']; kk = s['kind']; tolerance = .01 if metric == 'MAE' else .005
            assert s['protocol_sha256'] == sha('admet_v13_protocol.json')
            d = load_task(task); oof = baseline(task, d)
            candidate = pd.read_csv(OUT/task/(s['family']+'_oof.csv'))
            np.testing.assert_allclose(candidate.Y_true, oof.Y_true); np.testing.assert_array_equal(candidate.fold, oof.fold)
            gains = []
            for fold in range(3):
                mask = oof.fold.eq(fold); y = oof.loc[mask, 'Y_true'].to_numpy()
                b = oof.loc[mask, 'Y_pred'].to_numpy(); p = candidate.loc[mask, 'Y_pred'].to_numpy()
                before = metrics(kk, y, b); after = metrics(kk, y, (1-s['alpha'])*b+s['alpha']*p)
                gains.append(deltas(metric, before, after))
                assert gains[-1] >= -tolerance and secondary_ok(metric, before, after)
                it = np.flatnonzero(~mask); iv = np.flatnonzero(mask)
                domain = domain_check(d, it, iv, y, b, (1-s['alpha'])*b+s['alpha']*p, metric)
                assert domain['passed']
            assert np.mean(gains) > 0
            for split, filename in [('valid','validation_comparison.csv'), ('repeat','repeat_comparison.csv')]:
                frame = pd.read_csv(OUT/task/filename)
                before = metrics(kk, frame.Y_true, frame.baseline); after = metrics(kk, frame.Y_true, frame.blend)
                np.testing.assert_allclose(frame.blend, (1-s['alpha'])*frame.baseline+s['alpha']*frame.candidate, atol=1e-12)
                assert deltas(metric, before, after) >= (0 if split == 'valid' else -tolerance)
                assert secondary_ok(metric, before, after)
                n = len(d['frames']['train'])
                if split == 'valid':
                    it = np.arange(n); iv = np.arange(n, n+len(d['frames']['valid']))
                    np.testing.assert_allclose(frame.Y_true, d['frames']['valid'].Y)
                    frozen_valid = pd.read_csv(PRIOR/task/'validation_predictions.csv')
                    np.testing.assert_allclose(frame.baseline, frozen_valid.Y_pred, rtol=1e-12, atol=1e-12)
                else:
                    from sklearn.model_selection import GroupShuffleSplit
                    it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=692509).split(np.arange(n), groups=d['groups'][:n]))
                    np.testing.assert_array_equal(frame.row, iv)
                    np.testing.assert_allclose(frame.Y_true, d['frames']['train'].Y.to_numpy()[iv])
                domain = domain_check(d, it, iv, frame.Y_true.to_numpy(), frame.baseline.to_numpy(), frame.blend.to_numpy(), metric)
                assert domain['passed']
            assert all(x['passed'] for x in r['domains'])
            assert sha(folder/'model.joblib') == r['model_sha256']
            n = len(d['frames']['train'])+len(d['frames']['valid'])
            model = joblib.load(folder/'model.joblib')
            pred = predict(model, d, np.arange(n, len(d['smiles'])))
            np.testing.assert_allclose(pred, pd.read_csv(folder/'candidate_test.csv').Y, rtol=3e-6, atol=3e-6)
            np.testing.assert_allclose(result.Y, (1-s['alpha'])*pd.read_csv(oldfile).Y+s['alpha']*pred, rtol=3e-6, atol=3e-6)
            for audit in model.get('source_audit', []):
                assert audit['heldout_scaffold_overlap'] == audit['heldout_molecule_overlap'] == 0
            for audit_file in (OUT/task).glob('*source_audit.json'):
                for audit in json.loads(audit_file.read_text()):
                    assert audit['heldout_scaffold_overlap'] == audit['heldout_molecule_overlap'] == 0
        else:
            assert file.read_bytes() == oldfile.read_bytes()
            mask = new.task.eq(task)
            assert new.loc[mask, 'y_pred'].tolist() == old_upload.loc[mask, 'y_pred'].tolist()
        piece = new.loc[new.task == task]; pred = pd.to_numeric(piece.y_pred).to_numpy()
        np.testing.assert_allclose(pred, result.Y, rtol=1e-12, atol=1e-12)
        assert np.isfinite(pred).all()
        if piece.metric.iloc[0] in ['AUROC','AUPRC']:
            assert ((pred>=0)&(pred<=1)).all()
        checks.append(dict(task=task, accepted=r['accepted'], passed=True))
        emit(stage='v13_replay', task=task, passed=True)
    v.update(status='VERIFIED', independent_replay=True, task_checks=checks)
    save(ROOT/'independent_verification.json', v)


def report():
    from admet_v13_uncertainty import main as uncertainty
    uncertainty()
    routes = json.loads((ROOT/'routing.json').read_text()); rows = []
    lines = ['# v13 远程迭代结果', '',
        '平台实测基准为 v11 的 49.381036；v13 尚未提交，目标为 51.584929。', '',
        '本轮使用固定版本 MoLFormer 自监督预训练表示，比较表示核、表示与子结构混合核、带内部骨架早停的残差神经网络。仅下载无 ADMET 标签的预训练权重，未下载外部实验标签。', '',
        '|任务|指标|v11 valid|v13 最终 valid|采用方法|新权重|', '|---|---|---:|---:|---|---:|']
    for r in routes:
        task = r['task']; metric = get_metric(task)
        old = pd.read_csv(PRIOR/task/'validation_predictions.csv'); new = pd.read_csv(ROOT/task/'validation_predictions.csv')
        before = metrics(kind(metric), old.Y_true, old.Y_pred); after = metrics(kind(metric), new.Y_true, new.Y_pred)
        family = r['selection']['family'] if r['accepted'] else '保留 v11'; alpha = r['selection']['alpha'] if r['accepted'] else 0
        lines.append(f"|{task}|{metric}|{before[metric.lower()]:.6f}|{after[metric.lower()]:.6f}|{family}|{alpha}|")
        rows.append(dict(task=task, metric=metric, accepted=r['accepted'], family=family, alpha=alpha, reason=r['reason'],
                         **{'before_'+k:v for k,v in before.items()}, **{'after_'+k:v for k,v in after.items()}))
    lines += ['', '## 证据边界', '',
        '- 所有训练在授权远程目录完成。未使用外部实验标签，未读取盲测标签，未自动提交。',
        '- 本轮仅使用目标任务 train 拟合验证模型，最终发布模型使用该任务 train+valid；不使用其他任务或外部标签。',
        '- OOF、官方 valid 及额外切分均含历史复用，有选择偏差，不能保证隐藏集不退步或推算平台分数。',
        '- 条件骨架 bootstrap 区间见 v13_uncertainty.csv，只描述已选方案在复用验证集上的波动，未校正选型偏差，也不是平台总分区间。',
        '- 官方 valid 主指标要求不下降；各折及额外切分主指标容差为 MAE 1% 或排名指标 0.005；辅助指标容差为 MAE 3% 或排名指标 0.015。',
        '- 低相似度组小于 20 条或缺少两类时视为证据不足，本轮拒绝替换。主指标容差为 MAE 3% 或排名指标 0.01，另加辅助指标门槛。',
        '- 拒绝的任务保留 v11 原始预测字符串。新模型保存后在独立进程重载并核对推理和模板。',
        '- 沿用成功 v7 的 BOM、CRLF、六列及唯一表头；官网上传需采用已实测成功的整文件方式，刷新网页会恢复分片设置。', '']
    Path('admet_diagnostics/v13_result_report.md').write_text('\n'.join(lines), encoding='utf-8')
    pd.DataFrame(rows).to_csv('admet_diagnostics/v13_comparison.csv', index=False)
    keep = []
    for root in [ROOT, OUT]:
        keep += [f for f in root.rglob('*') if f.is_file() and f.suffix in ['.json','.csv'] and 'features' not in f.parts]
    keep += list(UPLOAD.parent.glob('*.csv')) if UPLOAD.parent.exists() else []
    keep += [f for f in Path('admet_diagnostics').glob('v13_*') if f.is_file()]
    keep += list(Path('logs').glob('v13*.log'))
    with tarfile.open('admet_v13_review.tar.gz', 'w:gz') as archive:
        for file in keep:
            archive.add(file, arcname=str(file), recursive=False)
    emit(stage='v13_packaged', archive_bytes=Path('admet_v13_review.tar.gz').stat().st_size,
         accepted=[r['task'] for r in routes if r['accepted']])


if __name__ == '__main__':
    remote_only(); parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['publish','verify','report']); args = parser.parse_args()
    {'publish':publish, 'verify':verify, 'report':report}[args.stage]()
