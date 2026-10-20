"""Conditional uncertainty audit after frozen selection; no extra model selection."""
import json
import numpy as np
import admet_v24_models as b
import admet_v37_run as v


def main():
    b.remote_only()
    d = b.build_data()
    report = json.loads((v.OUT / 'analysis.json').read_text())
    assert report['signature'] == v.signature()
    base = np.load(v.OUT / 'baseline_valid.npy')
    records = []
    rng = np.random.default_rng(370699)
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        k = d['tasks'].index(r['task'])
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        p = v.variants(np.load(v.OUT / r['task'] / 'valid/predictions.npz')['p'], r['metric'])[s['variant']]
        new = b.blend(r['metric'], base[vi], p, s['weight'])
        gain = b.delta(r['metric'], d['y'][vi], base[vi], new)
        np.testing.assert_allclose(gain, s['valid_gain'], rtol=1e-8, atol=1e-10)
        unique = np.unique(d['rg'][vi])
        buckets = [np.flatnonzero(d['rg'][vi] == group) for group in unique]
        gains = []
        for _ in range(300):
            ix = np.concatenate([buckets[j] for j in rng.integers(0, len(buckets), len(buckets))])
            g = b.delta(r['metric'], d['y'][vi[ix]], base[vi[ix]], new[ix])
            if np.isfinite(g):
                gains.append(g)
        held = []
        for fold in range(3):
            def utility(g):
                return np.mean([a for j, a in enumerate(g['cv_gains']) if j != fold])
            choices = [g for g in r['grid'] if utility(g) > .0005]
            best = max(choices, key=utility) if choices else None
            held.append(best['cv_gains'][fold] if best else 0.)
        record = dict(task=r['task'], metric=r['metric'], valid_before=b.metric(r['metric'], d['y'][vi], base[vi]),
                      valid_after=b.metric(r['metric'], d['y'][vi], new), valid_gain=gain,
                      post_selection_scaffold_bootstrap_95=np.quantile(gains, [.025, .975]).tolist(),
                      bootstrap_positive_fraction=float(np.mean(np.array(gains) > 0)),
                      held_fold_selection_gains=held, held_fold_mean=float(np.mean(held)),
                      cv_mean=s['cv_mean'], cv_gains=s['cv_gains'])
        records.append(record)
    b.save(v.OUT / 'uncertainty.json', dict(signature=v.signature(), tasks=records,
        limitation='Post-selection diagnostic on repeatedly reused validation and fold labels; intervals do not correct for model search. Held-fold selection is not fully nested. Official score unknown.'))


if __name__ == '__main__':
    main()
