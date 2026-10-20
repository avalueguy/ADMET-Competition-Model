"""Frozen-choice diagnostics, including a new repartition versus matched direct control."""
import json
import numpy as np
import torch
import admet_v24_models as b
import admet_v44_run as v

def main():
    b.remote_only()
    torch.backends.cuda.matmul.allow_tf32 = False
    d = b.build_data()
    report = json.loads((v.OUT/'analysis.json').read_text())
    assert report['signature']==v.signature() and report['status']=='EVALUATED'
    base = np.load(v.OUT/'baseline_valid.npy')
    old = np.load(v.OUT/'baseline_cv_proxy.npy')
    folds = np.load(b.OUT/'folds.npy')
    rng = np.random.default_rng(441020)
    records = []
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        k = d['tasks'].index(r['task'])
        vi = np.flatnonzero((d['tid']==k)&(d['split']==1))
        tr = np.flatnonzero((d['tid']==k)&(d['split']==0))
        vp = v.variants(np.load(v.OUT/r['task']/'valid/predictions.npz')['p'],r['metric'])
        cp = np.load(v.OUT/r['task']/'cv_predictions.npy')
        new = b.blend(r['metric'],base[vi],vp[s['variant']],s['weight'])
        clusters = np.unique(d['rg'][vi])
        buckets = [np.flatnonzero(d['rg'][vi]==g) for g in clusters]
        gains = []
        if len(buckets)>1:
            for _ in range(300):
                ix = np.concatenate([buckets[j] for j in rng.integers(0,len(buckets),len(buckets))])
                gain = b.delta(r['metric'],d['y'][vi[ix]],base[vi[ix]],new[ix])
                if np.isfinite(gain):gains.append(gain)
        control_cv,held = [],[]
        for fold in range(3):
            ix = tr[folds[tr]==fold]
            ctrl = b.blend(r['metric'],old[ix],cp[0,ix],s['weight'])
            pred = b.blend(r['metric'],old[ix],cp[s['variant'],ix],s['weight'])
            control_cv.append(b.delta(r['metric'],d['y'][ix],ctrl,pred))
            def utility(g):
                return np.mean([a for j,a in enumerate(g['adjusted_cv_gains']) if j!=fold])
            choices = [g for g in r['grid'] if utility(g)>.0005 and g['valid_gain']>-.02]
            best = max(choices,key=utility) if choices else None
            held.append(best['adjusted_cv_gains'][fold] if best else 0.)
        # Freeze a fresh, label-independent 20% scaffold repartition AFTER selection.
        allrows = np.flatnonzero((d['tid']==k)&(d['split']<2))
        groups = np.unique(d['rg'][allrows])
        rr = np.random.default_rng(441022+k)
        chosen = set(rr.permutation(groups)[:max(1,int(len(groups)*.2))])
        query = allrows[np.array([g in chosen for g in d['rg'][allrows]])]
        fitrows = allrows[np.array([g not in chosen for g in d['rg'][allrows]])]
        fresh = v.variants(v.fit(d,k,fitrows,query,v.OUT/r['task']/'fresh_scaffold'),r['metric'])
        fresh_gain = b.delta(r['metric'],d['y'][query],fresh[0],fresh[s['variant']])
        record = dict(task=r['task'],metric=r['metric'],variant=s['variant'],weight=s['weight'],
            valid_before=b.metric(r['metric'],d['y'][vi],base[vi]),valid_after=b.metric(r['metric'],d['y'][vi],new),
            valid_gain=s['valid_gain'],valid_clusters=len(clusters),largest_cluster=max(map(len,buckets)),
            bootstrap_95=np.quantile(gains,[.025,.975]).tolist() if gains else None,
            bootstrap_positive_fraction=float(np.mean(np.array(gains)>0)) if gains else None,
            matched_control_valid_gain=b.delta(r['metric'],d['y'][vi],b.blend(r['metric'],base[vi],vp[0],s['weight']),new),
            matched_control_cv_gains=control_cv,matched_control_cv_mean=float(np.mean(control_cv)),
            held_fold_selection_gains=held,held_fold_mean=float(np.mean(held)),
            fresh_fit_rows=len(fitrows),fresh_query_rows=len(query),fresh_seed=441022+k,fresh_group_overlap=0,
            fresh_control_metric=b.metric(r['metric'],d['y'][query],fresh[0]),
            fresh_selected_metric=b.metric(r['metric'],d['y'][query],fresh[s['variant']]),
            fresh_control_gain=fresh_gain,fresh_no_selection_changes=True,
            cv_gains=s['cv_gains'],cv_mean=s['cv_mean'],adjusted_cv_mean=s['adjusted_cv_mean'],
            valid_cyclic_fraction=r['valid_cyclic_fraction'],test_cyclic_fraction=r['test_cyclic_fraction'])
        records.append(record)
        b.save(v.OUT/'uncertainty.json',dict(signature=v.signature(),status='IN_PROGRESS',tasks=records))
        b.emit('v44_diagnostic',task=r['task'],fresh_control_gain=fresh_gain,held_fold_mean=record['held_fold_mean'])
    b.save(v.OUT/'uncertainty.json',dict(signature=v.signature(),status='COMPLETE',tasks=records,
        limitation='Post-selection reused validation/CV. Fresh repartition uses previously used labels, not independent external data; comparator direct control only. No official score forecast.'))

if __name__=='__main__':
    main()
