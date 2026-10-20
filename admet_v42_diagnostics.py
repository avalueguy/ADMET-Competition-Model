"""Post-selection uncertainty audit for the frozen combined release."""
import json
import numpy as np
import admet_v24_models as b
import admet_v40_run as tree
import admet_v42_run as v


def main():
    b.remote_only()
    d = b.build_data()
    report = json.loads((v.OUT / 'analysis.json').read_text())
    assert report['signature'] == v.signature()
    base = np.load(tree.OUT / 'baseline_valid.npy')
    rng = np.random.default_rng(420810)
    records = []
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        k = d['tasks'].index(r['task'])
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        values = []
        for source in s['sources']:
            p = v.source_predictions(r['task'], source['version'], source['variant'], r['metric'], 'valid')
            if s['name'] == 'complement_equal' and r['metric'] != 'MAE':
                p = b.rank(p)
            values.append(source['weight']*p)
        new = b.blend(r['metric'], base[vi], np.sum(values, axis=0), s['weight'])
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
        records.append(dict(task=r['task'], metric=r['metric'], valid_gain=gain,
            valid_before=b.metric(r['metric'], d['y'][vi], base[vi]), valid_after=b.metric(r['metric'], d['y'][vi], new),
            valid_scaffold_clusters=len(buckets), largest_cluster_rows=max(map(len, buckets)),
            cluster_bootstrap_degenerate=len(buckets)<2,
            post_selection_scaffold_bootstrap_95=np.quantile(gains, [.025, .975]).tolist(),
            bootstrap_positive_fraction=float(np.mean(np.array(gains)>0)), held_fold_selection_gains=held,
            held_fold_mean=float(np.mean(held)), cv_mean=s['cv_mean'], cv_gains=s['cv_gains']))
    b.save(v.OUT / 'uncertainty.json', dict(signature=v.signature(), tasks=records,
        limitation='Post-selection, reused historical validation/folds. Intervals do not adjust for model search. Not nested CV; not official score points.'))


if __name__ == '__main__':
    main()
