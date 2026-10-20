"""Post-selection diagnostics including direct representation-versus-control comparisons."""
import json
import numpy as np
import admet_v24_models as b
import admet_v43_run as v


def main():
    b.remote_only()
    d = b.build_data()
    report = json.loads((v.OUT / 'analysis.json').read_text())
    assert report['signature'] == v.signature()
    base = np.load(v.OUT / 'baseline_valid.npy')
    old = np.load(v.OUT / 'baseline_cv_proxy.npy')
    folds = np.load(b.OUT / 'folds.npy')
    rng = np.random.default_rng(430811)
    records = []
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        k = d['tasks'].index(r['task'])
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        vp = v.variants(np.load(v.OUT / r['task'] / 'valid/predictions.npz')['p'], r['metric'])
        cp = np.load(v.OUT / r['task'] / 'cv_predictions.npy')
        new = b.blend(r['metric'], base[vi], vp[s['variant']], s['weight'])
        control = b.blend(r['metric'], base[vi], vp[0], s['weight'])
        unique = np.unique(d['rg'][vi])
        buckets = [np.flatnonzero(d['rg'][vi] == group) for group in unique]
        gains = []
        if len(buckets) > 1:
            for _ in range(300):
                ix = np.concatenate([buckets[j] for j in rng.integers(0, len(buckets), len(buckets))])
                gain = b.delta(r['metric'], d['y'][vi[ix]], base[vi[ix]], new[ix])
                if np.isfinite(gain):
                    gains.append(gain)
        controls, held = [], []
        for fold in range(3):
            ix = tr[folds[tr] == fold]
            p = b.blend(r['metric'], old[ix], cp[s['variant'], ix], s['weight'])
            c = b.blend(r['metric'], old[ix], cp[0, ix], s['weight'])
            controls.append(b.delta(r['metric'], d['y'][ix], c, p))
            def utility(g):
                return np.mean([a for j, a in enumerate(g['adjusted_cv_gains']) if j != fold])
            choices = [g for g in r['grid'] if utility(g) > .0005 and g['valid_gain'] > -.005]
            best = max(choices, key=utility) if choices else None
            held.append(best['adjusted_cv_gains'][fold] if best else 0.)
        records.append(dict(task=r['task'], metric=r['metric'], variant=s['variant'], weight=s['weight'],
            valid_before=b.metric(r['metric'], d['y'][vi], base[vi]), valid_after=b.metric(r['metric'], d['y'][vi], new),
            valid_gain=s['valid_gain'], valid_clusters=len(buckets), largest_cluster=max(map(len, buckets)),
            bootstrap_degenerate=len(buckets)<2,
            bootstrap_95=np.quantile(gains, [.025, .975]).tolist() if gains else None,
            bootstrap_positive_fraction=float(np.mean(np.array(gains)>0)) if gains else None,
            matched_control_valid_gain=b.delta(r['metric'], d['y'][vi], control, new),
            matched_control_cv_gains=controls, matched_control_cv_mean=float(np.mean(controls)),
            held_fold_selection_gains=held, held_fold_mean=float(np.mean(held)),
            cv_gains=s['cv_gains'], cv_mean=s['cv_mean'], domain_cv_gains=s['domain_cv_gains'],
            adjusted_cv_mean=s['adjusted_cv_mean'], valid_cyclic_fraction=r['valid_cyclic_fraction'],
            test_cyclic_fraction=r['test_cyclic_fraction']))
    b.save(v.OUT / 'uncertainty.json', dict(signature=v.signature(), tasks=records,
        limitation='Post-selection, reused valid and CV proxy; intervals do not correct selection bias; no official score forecast.'))


if __name__ == '__main__':
    main()
