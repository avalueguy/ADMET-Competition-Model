"""Expanded chemical views and metric-aligned GPU boosting over the scored v36."""
import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem.rdFingerprintGenerator import (GetMorganGenerator, GetRDKitFPGenerator,
                                               GetAtomPairGenerator, GetTopologicalTorsionGenerator)
from scipy.stats import rankdata
from xgboost import XGBClassifier, XGBRegressor, XGBRanker
import admet_v24_models as b
import admet_v33_run as n
import admet_v36_run as previous

ROOT = n.ROOT
OUT = ROOT / 'admet_model_outputs_v37'
P = json.loads((ROOT / 'admet_v37_protocol.json').read_text(encoding='utf-8'))


def signature():
    names = ['admet_v37_run.py', 'admet_v37_protocol.json', 'admet_model_outputs_v36/analysis.json']
    return hashlib.sha256(json.dumps(dict(previous=previous.signature(),
        files={name: b.sha(ROOT / name) for name in names}), sort_keys=True).encode()).hexdigest()


def baseline(d):
    valid = np.load(n.OUT / 'baseline_valid.npy').copy()
    cv = np.load(n.OUT / 'baseline_cv_proxy.npy').copy()
    folds = np.load(b.OUT / 'folds.npy')
    report = json.loads((previous.OUT / 'analysis.json').read_text())
    assert report['signature'] == previous.signature()
    for record in report['tasks']:
        sel = record['selection']
        if sel is None:
            continue
        k = d['tasks'].index(record['task'])
        m = record['metric']
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        vp = np.zeros(len(vi))
        cp = np.zeros(len(tr))
        for source in sel['sources']:
            v = previous.source_predictions(d, record['task'], source['version'], source['variant'], 'valid')
            c = np.load(previous.SOURCES[source['version']].OUT / record['task'] / 'cv_predictions.npy')[source['variant'], tr].copy()
            if sel['name'] == 'complement_equal' and m != 'MAE':
                v = b.rank(v)
                for fold in range(3):
                    active = folds[tr] == fold
                    c[active] = b.rank(c[active])
            vp += source['weight'] * v
            cp += source['weight'] * c
        merged = b.blend(m, valid[vi], vp, sel['weight'])
        np.testing.assert_allclose(b.delta(m, d['y'][vi], valid[vi], merged), sel['valid_gain'], rtol=1e-8, atol=1e-10)
        valid[vi] = merged
        for fold in range(3):
            active = folds[tr] == fold
            ix = tr[active]
            cv[ix] = b.blend(m, cv[ix], cp[active], sel['weight'])
    np.save(OUT / 'baseline_valid.npy', valid)
    np.save(OUT / 'baseline_cv_proxy.npy', cv)
    return valid, cv


def chemistry(d):
    dest = OUT / 'features'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'chemistry.npz'
    meta = dict(smiles_sha=hashlib.sha256(json.dumps(d['smiles']).encode()).hexdigest(),
                source_sha=b.sha(__file__), rdkit=rdBase.rdkitVersion, views=P['chemical_views'])
    if path.exists():
        assert json.loads((dest / 'chemistry.json').read_text()) == meta
        return np.load(path)['x'].copy()
    generators = [GetMorganGenerator(radius=3, fpSize=4096, includeChirality=True),
                  GetRDKitFPGenerator(fpSize=2048, maxPath=7),
                  GetAtomPairGenerator(fpSize=1024, includeChirality=True),
                  GetTopologicalTorsionGenerator(fpSize=1024, includeChirality=True)]
    x = np.zeros((len(d['smiles']), 8192), np.float32)
    invalid = 0
    for i, smiles in enumerate(d['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            continue
        parts = [g.GetCountFingerprintAsNumPy(mol) for g in generators]
        x[i] = np.log1p(np.concatenate(parts)).clip(0, 8)
        if i % 10000 == 0:
            b.emit('v37_chemistry', done=i, total=len(x))
    assert np.isfinite(x).all()
    np.savez_compressed(path, x=x)
    meta['invalid_molecules'] = invalid
    # The invariant cache schema excludes statistics.
    b.save(dest / 'chemistry.json', {key: value for key, value in meta.items() if key != 'invalid_molecules'})
    b.save(dest / 'chemistry_audit.json', meta)
    return x


def matrices(d, k, rows, query, splits, other, chemical):
    x, pre, _, tf, audit = n.features(d, k, rows, query, splits, other)
    unique = np.unique(d['mi'][rows])
    # Each chemical representation gets a fit-only variance quota.
    boundaries = [0, 4096, 6144, 7168, 8192]
    selected = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        var = np.var(chemical[unique, start:end], axis=0)
        selected.extend((start + np.argsort(var, kind='stable')[-P['fingerprint_columns'] // 4:]).tolist())
    selected = np.array(selected, int)
    out = {}
    for name, rr in [('fit', rows), ('query', query)]:
        chem = chemical[d['mi'][rr]][:, selected]
        plain = np.concatenate([x[name]['dense'], chem], axis=1)
        rich = np.concatenate([x[name]['trees'][1], chem], axis=1)
        out[name] = [plain, rich, rich]
    audit.update(expanded_fingerprint_columns=selected.tolist(), fingerprints_label_free=True,
                 fingerprint_selection_fit_only=True, dense_preprocessor_fit_only=True)
    return out, tf, audit


def model_type(metric, family):
    if family == 2 and metric != 'MAE':
        return XGBRanker
    return XGBClassifier if metric in ['AUROC', 'AUPRC'] else XGBRegressor


def params(metric, family, seed, depth, estimators):
    cfg = dict(n_estimators=estimators, max_depth=depth, learning_rate=.035, min_child_weight=4,
               subsample=.85, colsample_bytree=.75, reg_lambda=18., reg_alpha=.10,
               tree_method='hist', device='cuda', max_bin=128, n_jobs=3, random_state=seed)
    if family == 2:
        if metric == 'MAE':
            cfg.update(objective='reg:quantileerror', quantile_alpha=.5)
        else:
            cfg.update(objective='rank:map' if metric == 'AUPRC' else 'rank:pairwise',
                       lambdarank_pair_method='mean', lambdarank_num_pair_per_sample=8)
    else:
        cfg['objective'] = 'binary:logistic' if metric in ['AUROC', 'AUPRC'] else ('reg:absoluteerror' if metric == 'MAE' else 'reg:squarederror')
    return cfg


def predict_model(model, x, metric, family, tf):
    if isinstance(model, XGBClassifier):
        return model.predict_proba(x)[:, 1]
    pred = model.predict(x)
    return tf.decode(pred) if metric == 'MAE' else pred


def fit(d, k, rows, query, splits, other, chemical, dest, families=None):
    dest.mkdir(parents=True, exist_ok=True)
    families = list(range(3)) if families is None else families
    cache = dest / 'predictions.npz'
    sig = signature()
    if cache.exists():
        z = np.load(cache)
        assert str(z['signature']) == sig
        np.testing.assert_array_equal(z['fit'], rows)
        np.testing.assert_array_equal(z['query'], query)
        assert set(families).issubset(set(z['families']))
        return z['p'].copy()
    t0 = time.time()
    x, tf, audit = matrices(d, k, rows, query, splits, other, chemical)
    m = d['metrics'][k]
    cls = m in ['AUROC', 'AUPRC']
    y = tf.encode(d['y'][rows])
    _, inv, count = np.unique(d['canonical'][d['mi'][rows]], return_inverse=True, return_counts=True)
    weights = 1 / np.sqrt(count[inv])
    pred = np.full((3, len(query)), np.nan)
    for family in families:
        parts = []
        for seed, depth, estimators in zip(P['seeds'], P['depths'], P['estimators']):
            cfg = params(m, family, seed, depth, estimators)
            klass = model_type(m, family)
            model = klass(**cfg)
            if family == 2 and m != 'MAE':
                relevance = y.astype(np.int32) if cls else np.floor((rankdata(d['y'][rows]) - 1) / len(rows) * 32).astype(np.int32)
                model.fit(x['fit'][family], relevance, qid=np.zeros(len(rows), np.int32))
            else:
                # Unweighted class prevalence avoids the previous universal positive-weight assumption.
                model.fit(x['fit'][family], y, sample_weight=weights)
            path = dest / f'family{family}_{seed}.ubj'
            model.save_model(path)
            p = predict_model(model, x['query'][family], m, family, tf)
            restored = klass()
            restored.load_model(path)
            again = predict_model(restored, x['query'][family], m, family, tf)
            np.testing.assert_allclose(p, again, rtol=2e-6, atol=2e-6)
            parts.append(p)
        # Keep non-MAE seeds comparable while preserving their continuous ranking.
        pred[family] = np.mean(parts, axis=0) if m == 'MAE' else np.mean([b.rank(p) for p in parts], axis=0)
        b.emit('v37_family', task=d['tasks'][k], partition=str(dest.relative_to(OUT)), family=family, seconds=round(time.time()-t0))
    assert np.isfinite(pred[families]).all()
    np.savez_compressed(cache, p=pred, fit=rows, query=query, signature=sig, families=families)
    audit.update(signature=sig, task=d['tasks'][k], fit_rows=len(rows), query_rows=len(query),
                 families=families, seconds=time.time()-t0, model_reload_passed=True,
                 test_labels_used=False, local_gpu_training=False)
    b.save(dest / 'metadata.json', audit)
    return pred


def variant_sources():
    return [[0], [1], [2], [0, 1], [1, 2], [0, 1, 2]]


def variants(p, metric):
    values = p.copy() if metric == 'MAE' else np.stack([b.rank(x) for x in p])
    return np.stack([np.mean(values[ix], axis=0) for ix in variant_sources()])


def screen(d, chemical, other):
    base, cv = baseline(d)
    records = []
    for k, task in enumerate(d['tasks']):
        tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        m = d['metrics'][k]
        vp = variants(fit(d, k, tr, vi, [0], other, chemical, OUT / task / 'valid'), m)
        options = [dict(variant=v, weight=w, valid_gain=b.delta(m, d['y'][vi], base[vi], b.blend(m, base[vi], vp[v], w)))
                   for v in range(len(vp)) for w in P['weights']]
        eligible = [r for r in options if r['valid_gain'] > .0002]
        record = dict(task=task, metric=m, valid_baseline=b.metric(m, d['y'][vi], base[vi]),
                      valid_rows=len(vi), train_rows=len(tr), screen_gain=max(x['valid_gain'] for x in options),
                      shortlisted_variants=sorted(set(x['variant'] for x in eligible)), screen_grid=options)
        records.append(record)
        b.save(OUT / 'screen.json', dict(signature=signature(), status='IN_PROGRESS', tasks=records))
        b.emit('v37_screen', task=task, gain=record['screen_gain'], variants=record['shortlisted_variants'])
    # The complete shortlist is frozen before accessing scaffold-fold metrics.
    b.save(OUT / 'screen.json', dict(signature=signature(), status='FROZEN', tasks=records,
                                  validation_reused=True, official_score_prediction=None))


def evaluate(d, chemical, other):
    screen_report = json.loads((OUT / 'screen.json').read_text())
    assert screen_report['status'] == 'FROZEN' and screen_report['signature'] == signature()
    base = np.load(OUT / 'baseline_valid.npy')
    old = np.load(OUT / 'baseline_cv_proxy.npy')
    folds = np.load(b.OUT / 'folds.npy')
    records = []
    for r in screen_report['tasks']:
        k = d['tasks'].index(r['task'])
        task, m = r['task'], r['metric']
        tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        record = dict(**r, selection=None, grid=[])
        if r['shortlisted_variants']:
            families = sorted({family for v in r['shortlisted_variants'] for family in variant_sources()[v]})
            cp = np.full((6, len(d['y'])), np.nan)
            for fold in range(3):
                q = tr[folds[tr] == fold]
                rows = tr[folds[tr] != fold]
                p = fit(d, k, rows, q, [0], other, chemical, OUT / task / f'fold{fold}', families)
                cp[:, q] = variants(p, m)
            np.save(OUT / task / 'cv_predictions.npy', cp)
            vp = variants(np.load(OUT / task / 'valid/predictions.npz')['p'], m)
            grid = []
            for v in r['shortlisted_variants']:
                for w in P['weights']:
                    gains = []
                    for fold in range(3):
                        ix = tr[folds[tr] == fold]
                        gains.append(b.delta(m, d['y'][ix], old[ix], b.blend(m, old[ix], cp[v, ix], w)))
                    vg = b.delta(m, d['y'][vi], base[vi], b.blend(m, base[vi], vp[v], w))
                    obj = float(.45 * vg + .55 * np.mean(gains) - .04 * np.std(gains))
                    grid.append(dict(variant=v, families=variant_sources()[v], weight=w, valid_gain=vg,
                                     cv_gains=gains, cv_mean=float(np.mean(gains)), objective=obj,
                                     eligible=bool(vg > .0002 and np.mean(gains) > 0 and obj > .0005)))
            choices = [g for g in grid if g['eligible']]
            record.update(grid=grid, selection=max(choices, key=lambda x: x['objective']) if choices else None)
        records.append(record)
        b.save(OUT / 'analysis.json', dict(signature=signature(), status='IN_PROGRESS', tasks=records))
        b.emit('v37_selection', task=task, selection=record['selection'])
    b.save(OUT / 'analysis.json', dict(signature=signature(), status='EVALUATED', baseline='v36', baseline_score=P['baseline_score'],
                                      tasks=records, selected_tasks=[r['task'] for r in records if r['selection']],
                                      official_score_prediction=None, limitations=P['limitations']))


def final(d, chemical):
    report = json.loads((OUT / 'analysis.json').read_text())
    assert report['status'] == 'EVALUATED' and report['signature'] == signature()
    other = n.other_cache(d, [0, 1])
    compact = {}
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        k = d['tasks'].index(r['task'])
        rows = np.flatnonzero((d['tid'] == k) & (d['split'] < 2))
        q = np.flatnonzero((d['tid'] == k) & (d['split'] == 2))
        pred = fit(d, k, rows, q, [0, 1], other, chemical, OUT / r['task'] / 'final', s['families'])
        compact[r['task']] = {str(f): pred[f].tolist() for f in s['families']}
    b.save(OUT / 'test_predictions_compact.json', compact)


def export(d):
    report = json.loads((OUT / 'analysis.json').read_text())
    assert report['signature'] == signature()
    if not report['selected_tasks']:
        b.emit('v37_no_candidate')
        return
    source = next((ROOT / 'admet_submissions/v36').glob('*.csv'))
    assert b.sha(source) == P['baseline_sha256']
    def read(path):
        return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'), newline='')))
    rows = read(source)
    template = read(ROOT / 'admet_official/submission_template.csv')
    compact = json.loads((OUT / 'test_predictions_compact.json').read_text())
    changed = {}
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        pos = [i for i, x in enumerate(rows) if i > 0 and x[0] == r['task']]
        old = np.array([float(rows[i][4]) for i in pos])
        values = [np.array(compact[r['task']][str(f)]) for f in s['families']]
        if r['metric'] != 'MAE':
            values = [b.rank(x) for x in values]
        p = np.mean(values, axis=0)
        assert len(p) == len(old)
        merged = b.blend(r['metric'], old, p, s['weight'])
        for i, value in zip(pos, merged):
            rows[i][4] = repr(float(value))
        changed[r['task']] = int(np.sum(merged != old))
    assert changed and len(rows) == len(template) == 16380 and rows[0] == template[0]
    for row, t in zip(rows[1:], template[1:]):
        assert len(row) == 6 and row[:4] + row[5:] == t[:4] + t[5:] and np.isfinite(float(row[4]))
        if row[5] in ['AUROC', 'AUPRC']:
            assert 0 <= float(row[4]) <= 1
    dest = ROOT / 'admet_submissions/v37'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / '我从山北来带着兰花草-小分子药物预测技术创新赛-20261006.csv'
    buf = io.StringIO(newline='')
    csv.writer(buf, lineterminator='\r\n').writerows(rows)
    path.write_bytes(b'\xef\xbb\xbf' + buf.getvalue().encode('utf-8'))
    assert read(path) == rows
    b.save(OUT / 'source_manifest.json', {name: b.sha(ROOT / name) for name in
        ['admet_v37_run.py', 'admet_v37_protocol.json', 'admet_v37_remote_verify.py']})
    b.save(dest / 'verification.json', dict(status='EXPERIMENTAL_AWAITING_PLATFORM', signature=signature(), baseline='v36',
        baseline_score=P['baseline_score'], baseline_sha256=b.sha(source), sha256=b.sha(path), rows=16379, tasks=22,
        columns=6, changed_tasks=changed, bytes=path.stat().st_size, encoding='UTF-8 BOM', line_endings='CRLF',
        template_identity_exact=True, training_host=P['training_host'], local_gpu_training=False, automatically_uploaded=False,
        platform_score=None, limitations=P['limitations']))
    b.emit('v37_export', sha256=b.sha(path), changed_tasks=changed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'evaluate', 'final', 'export', 'all'])
    args = parser.parse_args()
    b.remote_only()
    assert ROOT == Path(P['remote_root'])
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    d = b.build_data()
    chemical = chemistry(d)
    if args.stage in ['screen', 'all']:
        screen(d, chemical, n.other_cache(d, [0]))
    if args.stage in ['evaluate', 'all']:
        evaluate(d, chemical, n.other_cache(d, [0]))
    if args.stage in ['final', 'all']:
        final(d, chemical)
    if args.stage in ['export', 'all']:
        export(d)
    b.save(OUT / f'{args.stage}_completed.json', dict(signature=signature(), completed=True))


if __name__ == '__main__':
    main()
