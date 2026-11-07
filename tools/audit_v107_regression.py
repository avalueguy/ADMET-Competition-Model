"""Audit a reported regression using frozen predictions; never open test labels."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from verify_submission import verify


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def table(path):
    return list(csv.DictReader(io.StringIO(path.read_bytes().decode('utf-8-sig'), newline='')))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    return ((end - counts + 1 + end) / 2)[inverse]


def auc(y, p):
    positive, negative = p[y == 1], p[y == 0]
    delta = positive[:, None] - negative[None, :]
    return float(np.mean((delta > 0) + .5 * (delta == 0)))


def main():
    release = read(ROOT / 'admet_model_outputs_v107/release.json')
    anchor = ROOT / read(ROOT / 'admet_model_outputs_v101/release.json')['file']
    candidate = ROOT / release['file']
    assert sha(anchor) == release['anchor_sha256']
    assert sha(candidate) == release['sha256']
    old_rows, new_rows = table(anchor), table(candidate)
    assert len(old_rows) == len(new_rows)
    changes = Counter()
    for a, b in zip(old_rows, new_rows):
        assert {k: v for k, v in a.items() if k != 'y_pred'} == {k: v for k, v in b.items() if k != 'y_pred'}
        if a['y_pred'] != b['y_pred']:
            changes[a['task']] += 1
    assert dict(changes) == {'herg': 130}
    fmt = verify(candidate, ROOT / 'admet_official/submission_template.csv')

    # Match saved global row indices by the documented train/valid/template order.
    labels = []
    tasks = list(dict.fromkeys(r['task'] for r in old_rows))
    for task in tasks:
        for part in ('train', 'valid'):
            path = ROOT / 'admet_competition_data/data_benchmark' / task / (part + '.csv')
            labels.extend(float(r['Y']) for r in table(path))
        labels.extend([float('nan')] * sum(r['task'] == task for r in old_rows))
    labels = np.asarray(labels)
    records = []
    for part in ('fold0', 'fold1', 'fold2', 'valid', 'final'):
        count = dict(np.load(ROOT / f'admet_model_outputs_v102/models/herg/{part}/hybrid_direct_c30.0/predictions.npz'))
        design = dict(np.load(ROOT / f'admet_model_outputs_v85/design/herg/{part}/physical/design.npz'))
        residual = dict(np.load(ROOT / f'admet_model_outputs_v85/models/herg/{part}/physical_a0.3/predictions.npz'))
        np.testing.assert_array_equal(count['query'], design['query'])
        np.testing.assert_array_equal(count['query'], residual['query'])
        p85 = 1 / (1 + np.exp(-(design['qoff'] + .25 * residual['delta'])))
        shape = []
        for seed in (984026, 984027):
            z = dict(np.load(ROOT / f'admet_model_outputs_v98/models/herg/{part}/shape_d5/direct/{seed}/predictions.npz'))
            np.testing.assert_array_equal(count['query'], z['query'])
            shape.append(z['values'][z['snapshots'].tolist().index(1200)])
        p98 = 1 / (1 + np.exp(-np.mean(shape, axis=0)))
        baseline = .5 * ranks(p85) / len(p85) + .5 * ranks(p98) / len(p98)
        mixed = .5 * ranks(baseline) / len(baseline) + .5 * ranks(count['p']) / len(baseline)
        n = len(baseline)
        pair_ix = np.triu_indices(n, 1)
        before = np.sign(baseline[:, None] - baseline[None, :])[pair_ix]
        after = np.sign(mixed[:, None] - mixed[None, :])[pair_ix]
        item = dict(part=part, rows=n, fit_rows=len(count['fit']),
                    rank_correlation=float(np.corrcoef(ranks(baseline), ranks(count['p']))[0, 1]),
                    pair_order_changes=int(np.sum(before != after)), total_pairs=len(before))
        if part == 'final':
            expected_old = np.array([float(r['y_pred']) for r in old_rows if r['task'] == 'herg'])
            expected_new = np.array([float(r['y_pred']) for r in new_rows if r['task'] == 'herg'])
            np.testing.assert_array_equal(baseline, expected_old)
            np.testing.assert_array_equal(mixed, expected_new)
            assert not np.isfinite(labels[count['query']]).any()
            item.update(v101_reconstruction_exact=True, v107_reconstruction_exact=True,
                        rank_percentile_mean_shift=float(np.mean(np.abs(ranks(mixed) - ranks(baseline))) / n),
                        rank_percentile_max_shift=float(np.max(np.abs(ranks(mixed) - ranks(baseline)))) / n)
        else:
            y = labels[count['query']]
            score0, score1 = auc(y, baseline), auc(y, mixed)
            expected = next(r for r in release['checks'] if r['part'] == part)
            np.testing.assert_allclose(score1, expected['score'], atol=1e-14, rtol=0)
            np.testing.assert_allclose(score1 - score0, expected['gain'], atol=1e-14, rtol=0)
            item.update(positive_rows=int(np.sum(y == 1)), negative_rows=int(np.sum(y == 0)),
                        baseline_auc=score0, candidate_auc=score1, gain=score1 - score0)
        records.append(item)
    diagnostics = read(ROOT / 'admet_model_outputs_v107/diagnostics.json')
    result = dict(status='AUDIT_COMPLETE', version='v107', baseline='v101',
                  baseline_official_score_user_reported=50.932, official_score_user_reported=50.909,
                  official_delta=-.023, uploaded_byte_hash_confirmed=False,
                  conditional_attribution='Only hERG predictions changed; attribution assumes the delivered files were submitted under the same evaluator.',
                  sha256=sha(candidate), anchor_sha256=sha(anchor), changed_tasks=dict(changes),
                  format_verification=fmt, frozen_prediction_reconstruction=records,
                  uncertainty=diagnostics['records'],
                  established_limitations=['Historical validation labels reused in many experiment rounds.',
                                           'Valid gain interval includes zero.',
                                           'Baseline CV starts from historical proxy predictions, not a newly refitted complete v101 pipeline.',
                                           'Four reported gains include the two folds used for selection.'],
                  unknown_causes=['Exact effect of hidden-test distribution shift versus selection overfit cannot be separated without test labels.',
                                  'Official per-task scores and total-score aggregation are unavailable.'],
                  recommendation='Withdraw v107 recommendation; use v101 as development anchor. No weight sweep selected on the official result.',
                  test_labels_read=False, new_predictions_written=False)
    output = ROOT / 'docs/v107_regression_audit_20261026.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
