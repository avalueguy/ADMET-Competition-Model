"""Explore v75's selected fixed predictions after negative platform feedback.

This is a descriptive audit, not a new holdout, model selection or an estimate
of platform gains. It uses official train/valid labels and never opens test CSVs.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from verify_submission import read


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blend(a, p, weight):
    return (1 - weight) * rankdata(a) / len(a) + weight * rankdata(p) / len(p)


def score(metric, y, p):
    return float((roc_auc_score if metric == 'AUROC' else average_precision_score)(y, p))


def main():
    out = ROOT / 'admet_model_outputs_v75'
    release, protocol = load(out / 'release.json'), load(ROOT / 'admet_v75_protocol.json')
    confirmation = load(out / 'confirmation.json')
    template = ROOT / 'admet_official/submission_template.csv'
    assert sha(template) == '8ee41589d2c98e7b93081c058b3f928ac09bea4b3ffc899fa13677051e9baa20'
    with template.open(encoding='utf-8-sig', newline='') as stream:
        official = list(csv.DictReader(stream))
    tasks = list(dict.fromkeys(row['task'] for row in official))
    smiles, lookup, mi, tid, splits, labels = [], {}, [], [], [], []
    sources = {template.relative_to(ROOT).as_posix(): sha(template)}
    for k, task in enumerate(tasks):
        for si, part in enumerate(('train', 'valid', 'test')):
            if part == 'test':
                # Submission template supplies test structures, not test labels.
                rows = [{'Drug': r['smiles']} for r in official if r['task'] == task]
            else:
                path = ROOT / 'admet_competition_data/data_benchmark' / task / (part + '.csv')
                sources[path.relative_to(ROOT).as_posix()] = sha(path)
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.DictReader(stream))
            for row in rows:
                sm = row['Drug']
                if sm not in lookup:
                    lookup[sm] = len(smiles)
                    smiles.append(sm)
                mi.append(lookup[sm]); tid.append(k); splits.append(si)
                labels.append(float(row['Y']) if si < 2 else np.nan)
    mi, tid, splits, labels = map(np.asarray, (mi, tid, splits, labels))
    schema = load(ROOT / 'admet_model_outputs_v67/features/official_chemistry.json')
    assert hashlib.sha256(json.dumps(smiles).encode()).hexdigest() == schema['smiles_sha256']
    with np.load(ROOT / 'admet_model_outputs_v67/features/official_chemistry.npz') as data:
        groups = data['groups'][mi]
    folds = np.load(ROOT / 'admet_model_outputs_v67/folds.npy')
    cv = np.load(ROOT / 'recovered_history/baseline_cv_proxy.npy')
    valid = np.load(ROOT / 'recovered_history/baseline_valid.npy')
    for name in ('baseline_cv_proxy.npy', 'baseline_valid.npy'):
        path = ROOT / 'recovered_history' / name
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    anchor = next((ROOT / 'admet_submissions/v73_v72').glob('*.csv'))
    assert sha(anchor) == release['anchor_sha256'] and sha(ROOT / release['file']) == release['sha256']
    old_rows, new_rows = read(anchor)[1:], read(ROOT / release['file'])[1:]
    records = []
    for k, task in enumerate(tasks):
        if task not in release['selected']:
            continue
        selected = release['selected'][task]
        choice, metric = selected['choice'], selected['metric']
        assert metric in ('AUROC', 'AUPRC')
        parts = []
        for part in protocol['confirmation_parts']:
            q = np.flatnonzero((tid == k) & (splits == (1 if part == 'valid' else 0)))
            if part != 'valid':
                q = q[folds[q] == int(part[-1])]
            folder = out / task / part

            def pred(path):
                sources[path.relative_to(ROOT).as_posix()] = sha(path)
                with np.load(path) as z:
                    assert str(z['signature']) == release['signature']
                    np.testing.assert_array_equal(z['query'], q)
                    return z['p'].copy()

            kernel = pred(folder / 'kernels/predictions.npz')[choice['kernel']]
            expert = kernel
            if choice['conditional']:
                pp = [pred(folder / 'conditional_tree/outer/predictions.npz')]
                pp += [pred(folder / f'conditional_tree/outer_{seed}/predictions.npz') for seed in protocol['final_seeds'][1:]]
                expert = blend(kernel, np.mean(pp, axis=0), choice['conditional'])
            prior = valid[q] if part == 'valid' else cv[q]
            mixed = blend(prior, expert, protocol['anchor_weight'])
            y = labels[q]
            gain = score(metric, y, mixed) - score(metric, y, prior)
            reference = next(r for r in confirmation['tasks'] if r['task'] == task)
            reference = next(r for r in reference['parts'] if r['part'] == part)
            np.testing.assert_allclose(gain, reference['proxy_gain'], atol=1e-12)
            _, inverse = np.unique(groups[q], return_inverse=True)
            blocks = [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]
            rng = np.random.default_rng(752100 + 10 * k + (part == 'valid'))
            gains = []
            for _ in range(1000):
                ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
                if len(np.unique(y[ix])) < 2:
                    continue
                gains.append(score(metric, y[ix], mixed[ix]) - score(metric, y[ix], prior[ix]))
            lo, hi = np.quantile(gains, [.025, .975])
            parts.append(dict(part=part, metric=metric, rows=len(q), scaffold_groups=len(blocks),
                              fixed_prediction_proxy_gain=gain,
                              descriptive_paired_group_bootstrap_95_percentile=[float(lo), float(hi)],
                              bootstrap_draws=len(gains), interval_includes_zero=bool(lo <= 0 <= hi)))
        a = np.array([float(r[4]) for r in old_rows if r[0] == task])
        p = np.array([float(r[4]) for r in new_rows if r[0] == task])
        displacement = abs(rankdata(a) - rankdata(p)) / len(a)
        records.append(dict(task=task, choice=choice['name'], parts=parts,
                            label_free_test_ranking_change=dict(rows=len(a), spearman=float(spearmanr(a, p).statistic),
                            median_absolute_percentile_displacement=float(np.median(displacement)),
                            percentile_95_displacement=float(np.quantile(displacement, .95)))))
        print(json.dumps(records[-1]), flush=True)
    result = dict(status='DESCRIPTIVE_POST_FEEDBACK_AUDIT_COMPLETE', source_sha256=sha(Path(__file__)),
                  input_hashes=sources, tasks=records, test_csv_opened=False, test_labels_used=False,
                  full_v73_oof_available=False, new_expert_weight=protocol['anchor_weight'],
                  statistical_limitations='Fixed selected predictions on historically reused labels; exploratory group bootstrap conditional on the chosen methods. Does not account for selection, model fitting or anchor identity mismatch. Not an unbiased significance test, unseen holdout or platform-gain estimate.',
                  recommendation='Withdraw v75 recommendation and restore v73; exact reproducible baseline evidence is required before another replacement, rather than retuning from this feedback.')
    destination = ROOT / 'docs/v75_post_feedback_audit.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
