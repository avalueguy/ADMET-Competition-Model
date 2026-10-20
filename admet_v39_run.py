"""Freeze one combined submission from independently evaluated remote model sources."""
import csv
import hashlib
import io
import json
import numpy as np
import admet_v24_models as b
import admet_v37_run as tree
import admet_v38_run as kernel
ROOT = tree.ROOT
OUT = ROOT / 'admet_model_outputs_v39'
P = json.loads((ROOT / 'admet_v39_protocol.json').read_text())
SOURCES = {37: tree, 38: kernel}


def signature():
    return hashlib.sha256(json.dumps(dict(sources={str(v): m.signature() for v, m in SOURCES.items()},
        files={name: b.sha(ROOT / name) for name in ['admet_v39_run.py', 'admet_v39_protocol.json']}), sort_keys=True).encode()).hexdigest()


def source_predictions(task, version, variant, metric, partition):
    module = SOURCES[version]
    z = np.load(module.OUT / task / partition / 'predictions.npz')
    assert str(z['signature']) == module.signature()
    return module.variants(z['p'], metric)[variant]


def evaluate(d):
    base = np.load(tree.OUT / 'baseline_valid.npy')
    old = np.load(tree.OUT / 'baseline_cv_proxy.npy')
    folds = np.load(b.OUT / 'folds.npy')
    reports = {v: json.loads((m.OUT / 'analysis.json').read_text()) for v, m in SOURCES.items()}
    for v, r in reports.items():
        assert r['status'] == 'EVALUATED' and r['signature'] == SOURCES[v].signature()
    records = []
    for k, task in enumerate(d['tasks']):
        metric = d['metrics'][k]
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        candidates = []
        for version, report in reports.items():
            prior = next((r for r in report['tasks'] if r['task'] == task), None)
            if prior is None or prior['selection'] is None:
                continue
            variant = prior['selection']['variant']
            vp = source_predictions(task, version, variant, metric, 'valid')
            cp = np.load(SOURCES[version].OUT / task / 'cv_predictions.npy')[variant, tr]
            candidates.append(dict(name=f'v{version}', sources=[dict(version=version, variant=variant, weight=1.)], valid=vp, cv=cp))
        if len(candidates) > 1:
            vv, cc = [], []
            for candidate in candidates:
                v, c = candidate['valid'].copy(), candidate['cv'].copy()
                if metric != 'MAE':
                    v = b.rank(v)
                    for fold in range(3):
                        active = folds[tr] == fold
                        c[active] = b.rank(c[active])
                vv.append(v)
                cc.append(c)
            sources = [dict(**{k: c['sources'][0][k] for k in ['version', 'variant']}, weight=1/len(candidates)) for c in candidates]
            candidates.append(dict(name='complement_equal', sources=sources, valid=np.mean(vv, axis=0), cv=np.mean(cc, axis=0)))
        grid = []
        for candidate in candidates:
            for weight in P['weights']:
                gains = []
                for fold in range(3):
                    active = folds[tr] == fold
                    ix = tr[active]
                    gains.append(b.delta(metric, d['y'][ix], old[ix], b.blend(metric, old[ix], candidate['cv'][active], weight)))
                vg = b.delta(metric, d['y'][vi], base[vi], b.blend(metric, base[vi], candidate['valid'], weight))
                obj = float(.45*vg + .55*np.mean(gains) - .04*np.std(gains))
                grid.append(dict(name=candidate['name'], sources=candidate['sources'], weight=weight, valid_gain=vg,
                    cv_gains=gains, cv_mean=float(np.mean(gains)), objective=obj,
                    eligible=bool(vg > .0002 and np.mean(gains) > 0 and obj > .0005)))
        choices = [g for g in grid if g['eligible']]
        s = max(choices, key=lambda x: x['objective']) if choices else None
        records.append(dict(task=task, metric=metric, valid_baseline=b.metric(metric, d['y'][vi], base[vi]), selection=s, grid=grid))
    b.save(OUT / 'analysis.json', dict(signature=signature(), status='EVALUATED', baseline='v36', baseline_score=P['baseline_score'],
        tasks=records, selected_tasks=[r['task'] for r in records if r['selection']], official_score_prediction=None, limitations=P['limitations']))


def export(d):
    report = json.loads((OUT / 'analysis.json').read_text())
    assert report['signature'] == signature()
    source = next((ROOT / 'admet_submissions/v36').glob('*.csv'))
    assert b.sha(source) == P['baseline_sha256']
    def read(path):
        return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'), newline='')))
    rows = read(source)
    template = read(ROOT / 'admet_official/submission_template.csv')
    compact, changed = {}, {}
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        task = r['task']
        pos = [i for i, row in enumerate(rows) if i > 0 and row[0] == task]
        old = np.array([float(rows[i][4]) for i in pos])
        values, rawparts = [], []
        for source in s['sources']:
            pred = source_predictions(task, source['version'], source['variant'], r['metric'], 'final')
            rawparts.append(dict(**source, predictions=pred.tolist()))
            if s['name'] == 'complement_equal' and r['metric'] != 'MAE':
                pred = b.rank(pred)
            values.append(source['weight'] * pred)
        compact[task] = rawparts
        p = np.sum(values, axis=0)
        assert len(p) == len(old)
        merged = b.blend(r['metric'], old, p, s['weight'])
        for i, value in zip(pos, merged):
            rows[i][4] = repr(float(value))
        changed[task] = int(np.sum(merged != old))
    assert changed and len(rows) == len(template) == 16380 and rows[0] == template[0]
    for row, target in zip(rows[1:], template[1:]):
        assert len(row) == 6 and row[:4]+row[5:] == target[:4]+target[5:] and np.isfinite(float(row[4]))
        if row[5] in ['AUROC', 'AUPRC']:
            assert 0 <= float(row[4]) <= 1
    dest = ROOT / 'admet_submissions/v39'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / '我从山北来带着兰花草-小分子药物预测技术创新赛-20261006.csv'
    buf = io.StringIO(newline='')
    csv.writer(buf, lineterminator='\r\n').writerows(rows)
    path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'))
    assert read(path) == rows
    b.save(OUT / 'test_predictions_compact.json', compact)
    names = ['admet_v39_run.py', 'admet_v39_protocol.json', 'admet_v37_run.py', 'admet_v37_protocol.json',
             'admet_v37_resume.py', 'admet_v38_run.py', 'admet_v38_protocol.json', 'admet_v38_remote_verify.py']
    b.save(OUT / 'source_manifest.json', {name: b.sha(ROOT / name) for name in names})
    replay = {}
    for v, module in SOURCES.items():
        verify = json.loads((module.OUT / 'model_replay.json').read_text())
        assert verify['signature'] == module.signature()
        replay[str(v)] = verify
    b.save(OUT / 'model_replay.json', dict(status='SOURCE_MODEL_REPLAYS_VERIFIED', signature=signature(), sources=replay))
    b.save(dest / 'verification.json', dict(status='EXPERIMENTAL_AWAITING_PLATFORM', signature=signature(), baseline='v36',
        baseline_score=P['baseline_score'], baseline_sha256=P['baseline_sha256'], sha256=b.sha(path), rows=16379,
        tasks=22, columns=6, changed_tasks=changed, bytes=path.stat().st_size, encoding='UTF-8 BOM', line_endings='CRLF',
        template_identity_exact=True, training_host=P['training_host'], local_gpu_training=False, automatically_uploaded=False,
        platform_score=None, limitations=P['limitations']))
    b.emit('v39_export', sha256=b.sha(path), changed_tasks=changed)


if __name__ == '__main__':
    b.remote_only()
    OUT.mkdir(parents=True, exist_ok=True)
    d = b.build_data()
    evaluate(d)
    export(d)
