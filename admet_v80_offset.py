"""v80: fixed conditional CYP2C9 residual; no test labels or external data.

Run `python admet_v80_offset.py meta` then `python admet_v80_offset.py export`.
Historical proxy cross-fitting is diagnostic, not a full nested v73 evaluation.
Rank features are computed on the specified complete fit/query cohorts.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '4'
import argparse
import csv
import io
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
import admet_v77_ranking as data
from repack_upload_template import field_spans
from verify_submission import verify

OUT = ROOT / 'admet_model_outputs_v80_final'
TASK = 'cyp2c9_veith'
ANCHOR_SHA = 'beab85b972b336526a30c8baa6d181552927f0d004ac6360f42ecdbd96399763'
FOLD_SHA = '3a24d8a1359df47c4197bff62a80809bbdc42f53db22c5dd1f478db2a95a4b98'

def spec():
    s = json.loads((ROOT / 'admet_v80_protocol.json').read_text(encoding='utf-8'))
    assert s['task'] == TASK and s['anchor_sha256'] == ANCHOR_SHA
    assert s['trees'] == 300 and s['depth'] == 4 and s['blend_weight'] == 1
    assert s['seeds'] == [802026, 802027, 802028]
    return s

def read_csv(path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'), newline='')))

def anchor():
    p = next((ROOT / 'admet_submissions/v73_v72').glob('*.csv'))
    assert data.sha(p) == ANCHOR_SHA
    return p

def ranklog(p):
    p = np.asarray(p)
    assert np.isfinite(p).all() and len(p) > 0
    return logit(np.clip((rankdata(p) - .5) / len(p), .002, .998))

def auxiliary(d, k, include_valid=False):
    """Only other tasks' permitted measurements; target value AND mask absent."""
    cols = [j for j in range(len(d['tasks'])) if j != k]
    source = np.flatnonzero((d['split'] < (2 if include_valid else 1)) & (d['tid'] != k))
    assert np.isfinite(d['y'][source]).all() and np.all(d['split'][source] < 2)
    known = {}
    for i in source:
        known.setdefault((str(d['canonical'][i]), int(d['tid'][i])), []).append(float(d['y'][i]))
    aux = np.full((len(d['y']), 2 * len(cols)), np.nan, np.float32)
    for i in np.flatnonzero(d['tid'] == k):
        for j, t in enumerate(cols):
            values = known.get((str(d['canonical'][i]), t))
            if values:
                aux[i, j] = np.median(values)
    aux[:, len(cols):] = np.isfinite(aux[:, :len(cols)])
    return aux, source, cols

def calibration(y, z):
    x = np.c_[np.ones(len(z)), z]
    def objective(theta):
        eta = x @ theta
        residual = expit(eta) - y
        return ((np.logaddexp(0, eta) - y * eta).mean() + .001 * theta[1] ** 2,
                x.T @ residual / len(y) + [0., .002 * theta[1]])
    result = minimize(objective, [0., 1.], jac=True, method='L-BFGS-B',
                      bounds=[(None, None), (.01, 10.)])
    assert result.success, result.message
    return result.x

def model_config(seed):
    s = spec()
    return dict(iterations=s['trees'], depth=s['depth'], learning_rate=s['learning_rate'],
                l2_leaf_reg=s['l2_leaf_reg'], loss_function='Logloss', random_seed=seed,
                thread_count=4, verbose=False, allow_writing_files=False,
                bootstrap_type='Bernoulli', subsample=.85)

def fit_residual(d, fit, query, p, qp, aux, folder):
    folder.mkdir(parents=True, exist_ok=True)
    assert not set(fit) & set(query) and np.all(d['split'][fit] < 2)
    z, qz = ranklog(p), ranklog(qp)
    beta = calibration(d['y'][fit], z)
    offset, qoff = beta[0] + beta[1] * z, beta[0] + beta[1] * qz
    x, qx = np.c_[aux[fit], z].astype(np.float32), np.c_[aux[query], qz].astype(np.float32)
    members, model_hashes = [], {}
    for seed in spec()['seeds']:
        net = CatBoostClassifier(**model_config(seed))
        net.fit(Pool(x, d['y'][fit], baseline=offset))
        members.append(net.predict(qx, prediction_type='RawFormulaVal'))
        path = folder / f'model_{seed}.cbm'
        net.save_model(str(path))
        model_hashes[path.name] = data.sha(path)
    members = np.stack(members)
    raw = qoff + members.mean(axis=0)
    probability = expit(raw)
    np.savez_compressed(folder / 'predictions.npz', fit=fit, query=query, base_fit=p,
        base_query=qp, beta=beta, x=x, qx=qx, offset=offset, qoff=qoff,
        residual_members=members, raw=raw, p=probability)
    data.save(folder / 'metadata.json', dict(model_hashes=model_hashes,
        state_sha256=data.sha(folder / 'predictions.npz'), seeds=spec()['seeds'],
        config=model_config(spec()['seeds'][0]), fit_rows=len(fit), query_rows=len(query),
        rank_policy='Complete specified fit and query cohorts separately; clip .002/.998; average ties',
        signature=data.sha(__file__)))
    return probability

def meta_check(d):
    k = d['tasks'].index(TASK)
    foldpath = ROOT / 'recovered_history/legacy_v24_folds.npy'
    assert data.sha(foldpath) == FOLD_SHA
    folds = np.load(foldpath)
    tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
    va = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
    aux, source, cols = auxiliary(d, k)
    # Preserve development probe's full-train rank feature definition exactly.
    z, vz = ranklog(d['proxy_cv'][tr]), ranklog(d['proxy_valid'][va])
    x = np.c_[aux[tr], z].astype(np.float32)
    vx = np.c_[aux[va], vz].astype(np.float32)
    records = []
    out = ROOT / 'admet_model_outputs_v80_meta'
    for part in ['fold0', 'fold1', 'fold2', 'valid']:
        if part == 'valid':
            a = np.arange(len(tr)); q = va; qx = vx; qz = vz; old = d['proxy_valid'][q]
        else:
            f = int(part[-1]); a = np.flatnonzero(folds[tr] != f)
            local_q = np.flatnonzero(folds[tr] == f); q = tr[local_q]
            qx = x[local_q]; qz = z[local_q]; old = d['proxy_cv'][q]
        beta = calibration(d['y'][tr[a]], z[a]); members = []
        folder = out / part; folder.mkdir(parents=True, exist_ok=True)
        for seed in spec()['seeds']:
            net = CatBoostClassifier(**model_config(seed))
            net.fit(Pool(x[a], d['y'][tr[a]], baseline=beta[0] + beta[1] * z[a]))
            members.append(net.predict(qx, prediction_type='RawFormulaVal'))
            net.save_model(str(folder / f'model_{seed}.cbm'))
        members = np.stack(members); qoff = beta[0] + beta[1] * qz
        raw = qoff + members.mean(axis=0)
        np.savez_compressed(folder / 'predictions.npz', fit=tr[a], query=q, p=raw,
                            qx=qx, qoff=qoff, beta=beta, members=members)
        base, new = float(ap(d['y'][q], old)), float(ap(d['y'][q], raw))
        records.append(dict(part=part, rows=len(q), baseline_ap=base,
                            candidate_ap=new, gain=new-base, accepted=new > base))
        print(json.dumps(records[-1]), flush=True)
    result = dict(status='COMPLETE', accepted=all(r['accepted'] for r in records),
        records=records, seeds=spec()['seeds'], target_auxiliary_excluded=True,
        auxiliary_source_splits=[0], full_nested_v73_ensemble=False,
        limitation='Historical proxy meta crossfit; reused development labels; base proxy training scopes overlap meta folds.')
    data.save(out / 'confirmation.json', result)
    assert result['accepted'], 'Frozen three-seed meta diagnostic failed; do not export.'

def export(d):
    for relative in ['admet_model_outputs_v80_nested/confirmation.json',
                     'admet_model_outputs_v80_meta/confirmation.json']:
        assert json.loads((ROOT / relative).read_text())['accepted'], relative
    k = d['tasks'].index(TASK)
    rows = read_csv(anchor())
    tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
    va = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
    fit = np.r_[tr, va]
    query = np.flatnonzero((d['tid'] == k) & (d['split'] == 2))
    positions = [i for i, row in enumerate(rows) if i and row[0] == TASK]
    assert len(query) == len(positions) == 2419
    assert [rows[i][3] for i in positions] == [r['smiles'] for r in d['template'] if r['task'] == TASK]
    p = np.r_[d['proxy_cv'][tr], d['proxy_valid'][va]]
    qp = np.array([float(rows[i][4]) for i in positions])
    aux, source, cols = auxiliary(d, k, include_valid=True)
    folder = OUT / TASK
    prediction = fit_residual(d, fit, query, p, qp, aux, folder)
    np.save(folder / 'auxiliary_source_rows.npy', source)
    assert np.isfinite(prediction).all() and np.all((prediction > 0) & (prediction < 1))
    for i, value in zip(positions, prediction):
        rows[i][4] = repr(float(value))
    template = ROOT / 'admet_official/submission_template.csv'
    raw = template.read_bytes(); spans = field_spans(raw); chunks = []; cursor = 0
    for i, row in enumerate(rows[1:], 1):
        lo, hi = spans[i][4]; value = row[4].encode('ascii')
        assert not any(c in value for c in b',\r\n"')
        chunks.extend((raw[cursor:lo], value)); cursor = hi
    chunks.append(raw[cursor:])
    dest = ROOT / 'admet_submissions/v80_cyp2c9_residual'
    dest.mkdir(parents=True, exist_ok=True)
    filename = anchor().name.replace('20261021', '20261023')
    file = dest / filename
    file.write_bytes(b''.join(chunks))
    audit = verify(file, template)
    file.with_suffix('.verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    inputs = dict(d['sources'])
    for path in [anchor(), ROOT/'admet_v80_protocol.json', ROOT/'admet_v80_offset.py',
                 ROOT/'admet_v80_nested.py', ROOT/'recovered_history/legacy_v24_folds.npy',
                 ROOT/'admet_model_outputs_v80_nested/confirmation.json',
                 ROOT/'admet_model_outputs_v80_meta/confirmation.json']:
        inputs[path.relative_to(ROOT).as_posix()] = data.sha(path)
    data.save(OUT/'release.json', dict(status='CANDIDATE_PENDING_INDEPENDENT_REPLAY', version='v80',
        file=file.relative_to(ROOT).as_posix(), sha256=data.sha(file), anchor_sha256=ANCHOR_SHA,
        changed_tasks={TASK:len(query)}, unchanged_tasks=21, seeds=spec()['seeds'],
        feature_tasks=[d['tasks'][j] for j in cols], auxiliary_source_splits=[0,1],
        auxiliary_source_rows=len(source), target_auxiliary_value_and_mask_absent=True,
        input_hashes=inputs, test_csv_opened=False, test_labels_used=False, external_data=False,
        official_score=None, automatically_uploaded=False, verification=audit,
        limitations=['Official score requires platform submission.',
            'Nested control retrains a structure base, not the full historical v73 ensemble.',
            'All development labels have been reused historically.',
            'Conditional evaluation permits other assays of held-target molecules from allowed source splits.',
            'Rank transform depends on the specified full query cohort.']))
    print(json.dumps(dict(status='EXPORTED', file=str(file), sha256=data.sha(file)), ensure_ascii=False), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['meta','export'])
    args = parser.parse_args()
    d = data.load_data()
    (meta_check if args.stage == 'meta' else export)(d)

if __name__ == '__main__':
    main()
