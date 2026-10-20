"""Post-selection diagnostics; never changes frozen model or weight choices."""
import json
from pathlib import Path
import numpy as np
import admet_v24_models as b
import admet_v46_run as v

ROOT=v.ROOT
OUT=v.OUT

def assess(d,k,ix,a,c):
    if len(ix)<8 or len(np.unique(d['y'][ix]))<2:return None
    return float(b.delta(d['metrics'][k],d['y'][ix],a,c))

def main():
    b.remote_only()
    d=b.build_data()
    folds=np.load(b.OUT/'folds.npy')
    base=np.load(OUT/'baseline_valid.npy')
    old=np.load(OUT/'baseline_cv_proxy.npy')
    report=json.loads((OUT/'analysis.json').read_text())
    assert report['status']=='FROZEN' and report['signature']==v.signature()
    result=[]
    for r in report['tasks']:
        s=r['selection']
        if not s:continue
        k=d['tasks'].index(r['task']);m=r['metric']
        vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        tr=np.flatnonzero((d['tid']==k)&(d['split']==0))
        te=np.flatnonzero((d['tid']==k)&(d['split']==2))
        vp=np.load(OUT/r['task']/'valid_predictions.npy')
        cp=np.load(OUT/r['task']/'cv_predictions.npy')
        chosen=s['variant']
        corresponding_original=0 if chosen==2 else 1
        if chosen in [4,5]:
            control_v=.5*(vp[0]+vp[1])
            control_c=.5*(cp[0]+cp[1])
        else:
            control_v=vp[corresponding_original]
            control_c=cp[corresponding_original]
        newv=b.blend(m,base[vi],vp[chosen],s['weight'])
        ctrlv=b.blend(m,base[vi],control_v,s['weight'])
        ctrl_gains=[]
        lofo=[]
        for f in range(3):
            ix=tr[folds[tr]==f]
            control=b.blend(m,old[ix],control_c[ix],s['weight'])
            new=b.blend(m,old[ix],cp[chosen,ix],s['weight'])
            ctrl_gains.append(b.delta(m,d['y'][ix],control,new))
            # Re-select using only the other folds. The held-out fold is used
            # here solely to diagnose instability in the original search.
            choices=[]
            for row in r['grid']:
                gg=[g for j,g in enumerate(row['cv_gains']) if j!=f]
                obj=.2*row['valid_gain']+.8*np.mean(gg)-.05*np.std(gg)
                if np.mean(gg)>.0005 and obj>.0005:
                    choices.append((obj,row))
            choice=max(choices,key=lambda x:x[0])[1] if choices else None
            lofo.append(dict(fold=f,gain=choice['cv_gains'][f] if choice else 0.,
                selected_variant=choice['variant'] if choice else None,
                selected_weight=choice['weight'] if choice else 0.))
        rng=np.random.default_rng(461015+k)
        groups=d['rg'][vi]
        unique=np.unique(groups)
        buckets=[np.flatnonzero(groups==g) for g in unique]
        samples=[]
        for _ in range(300):
            ix=np.concatenate([buckets[i] for i in rng.integers(len(buckets),size=len(buckets))])
            gain=assess(d,k,vi[ix],base[vi][ix],newv[ix])
            if gain is not None:samples.append(gain)
        cyclic=np.array([not str(g).startswith('ACYCLIC:') for g in d['groups']])
        domain=[]
        for f in range(3):
            ix=tr[folds[tr]==f]
            new=b.blend(m,old[ix],cp[chosen,ix],s['weight'])
            for ring in [False,True]:
                active=cyclic[d['mi'][ix]]==ring
                gain=assess(d,k,ix[active],old[ix][active],new[active])
                domain.append(dict(fold=f,cyclic=ring,rows=int(sum(active)),gain=gain))
        result.append(dict(task=r['task'],metric=m,selection=s,
            fine_tune_selected=chosen>=2,
            fine_tune_vs_original_valid_gain=b.delta(m,d['y'][vi],ctrlv,newv) if chosen>=2 else 0.,
            fine_tune_vs_original_cv_gains=ctrl_gains if chosen>=2 else [0.,0.,0.],
            valid_scaffold_count=len(unique),
            valid_bootstrap_interval=np.quantile(samples,[.025,.975]).tolist() if samples else None,
            leave_one_fold_selection=lofo,
            leave_one_fold_mean_gain=float(np.mean([a['gain'] for a in lofo])),
            domain_cv_gains=domain,valid_cyclic_fraction=float(cyclic[d['mi'][vi]].mean()),
            test_cyclic_fraction=float(cyclic[d['mi'][te]].mean())))
    b.save(OUT/'diagnostics.json',dict(signature=v.signature(),tasks=result,
        frozen_choices_unchanged=True,limitations='Post-selection diagnostics on historically reused labels; not independent validation. Bootstrap intervals do not correct model search bias.'))
    print(json.dumps([dict(task=r['task'],variant=r['selection']['name'],
        cv=r['selection']['cv_mean'],lofo=r['leave_one_fold_mean_gain'],
        fine_tune_delta=np.mean(r['fine_tune_vs_original_cv_gains'])) for r in result],indent=2))

if __name__=='__main__':main()
