"""Freeze v74 choices before loading fold2 or official-valid predictions."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v74_specialists as v

VARIANTS={'graph':[0],'fingerprint':[1],'hybrid':[2],'fp_hybrid':[1,2],'graph_hybrid':[0,2],'all':[0,1,2]}

def expert(task,part,choice):
    folder=v.OUT/task/part
    z=np.load(folder/'outer/predictions.npz');t=np.load(folder/'tree/predictions.npz')
    np.testing.assert_array_equal(z['fit'],t['fit']);np.testing.assert_array_equal(z['query'],t['query'])
    ps=[z['p']]
    for seed in v.P['final_seeds'][1:]:
        path=folder/f'outer_{seed}/predictions.npz'
        if path.exists():
            a=np.load(path);np.testing.assert_array_equal(a['fit'],z['fit']);np.testing.assert_array_equal(a['query'],z['query']);ps.append(a['p'])
    p=np.mean(ps,axis=0)[choice['heads']].mean(0)
    return (1-choice['tree_weight'])*p+choice['tree_weight']*t['p'],t['p'],z['query']

def select(d):
    dest=v.OUT/'frozen_selection.json'
    if dest.exists():
        x=json.loads(dest.read_text(encoding='utf-8'));assert x['signature']==v.signature();return x
    assert not any((v.OUT/t/p/'outer/predictions.npz').exists() for t in v.P['tasks'] for p in ('fold2','valid')),'Selection must freeze before confirmation is trained'
    cp=np.load(ROOT/'recovered_history/baseline_cv_proxy.npy');records=[]
    for task in v.P['tasks']:
        candidates=[]
        for name,heads in VARIANTS.items():
            for tree_weight in (0.,.5):
                c=dict(name=name,heads=heads,tree_weight=tree_weight)
                gains=[];proxy=[];scores=[]
                for part in ('fold0','fold1'):
                    p,tree,q=expert(task,part,c);y=d['y'][q]
                    gains.append(b.gain('MAE',y,tree,p));scores.append(b.metric_score('MAE',y,p))
                    proxy.append(b.gain('MAE',y,cp[q],.5*cp[q]+.5*p))
                c.update(same_rows_tree_gains=gains,proxy_blend_gains=proxy,scores=scores,objective=float(np.mean(gains)-.25*np.std(gains)))
                candidates.append(c)
        eligible=[c for c in candidates if min(c['same_rows_tree_gains'])>0 and np.mean(c['proxy_blend_gains'])>0 and min(c['proxy_blend_gains'])>-.005]
        winner=max(eligible,key=lambda c:c['objective']) if eligible else None
        if winner is not None:winner=dict(**winner,anchor_weight=.5)
        records.append(dict(task=task,selection=winner,grid=candidates))
        print(json.dumps(dict(task=task,selection=winner)),flush=True)
    result=dict(status='FROZEN_BEFORE_CONFIRMATION',signature=v.signature(),tasks=records,
        selection_source_sha256=b.sha(__file__),confirmation_runner_sha256=b.sha(ROOT/'tools/run_v74_confirmation.py'),
        confirmation_parts=['fold2','valid'],anchor='v73',anchor_weight_fixed=.5,
        confirmation_gate='Both parts must improve the same-fit-row tree comparator; mean relative MAE gain > 0.002. Frozen half-anchor mixture must improve historical proxy on each part. No retuning on confirmation.',
        historical_proxy_is_exact_anchor=False,test_labels_used=False,
        limitation='Historical arrays are only proxy diagnostics, trained on different historical folds. Development labels reused historically; confirmation is not an unseen external test.')
    b.save(dest,result);return result

def confirm(d):
    frozen=json.loads((v.OUT/'frozen_selection.json').read_text(encoding='utf-8'));assert frozen['signature']==v.signature()
    cp=np.load(ROOT/'recovered_history/baseline_cv_proxy.npy');vp=np.load(ROOT/'recovered_history/baseline_valid.npy');records=[]
    for r in frozen['tasks']:
        c=r['selection'];task=r['task']
        if c is None:records.append(dict(task=task,accepted=False,reason='Selection-fold gate failed'));continue
        parts=[]
        for part in ('fold2','valid'):
            assert all((v.OUT/task/part/f'outer_{seed}/predictions.npz').exists() for seed in v.P['final_seeds'][1:]),'Confirm the final ensemble seed count'
            p,tree,q=expert(task,part,c);y=d['y'][q];old=vp[q] if part=='valid' else cp[q]
            parts.append(dict(part=part,n=len(q),tree_mae=b.metric_score('MAE',y,tree),expert_mae=b.metric_score('MAE',y,p),same_rows_tree_gain=b.gain('MAE',y,tree,p),
                historical_proxy_mae=b.metric_score('MAE',y,old),half_anchor_proxy_mae=b.metric_score('MAE',y,.5*old+.5*p),proxy_gain=b.gain('MAE',y,old,.5*old+.5*p)))
        accepted=bool(min(a['same_rows_tree_gain'] for a in parts)>0 and np.mean([a['same_rows_tree_gain'] for a in parts])>.002 and min(a['proxy_gain'] for a in parts)>0)
        records.append(dict(task=task,selection=c,accepted=accepted,parts=parts))
        print(json.dumps(records[-1]),flush=True)
    result=dict(status='CONFIRMATION_COMPLETE',signature=v.signature(),frozen_selection_sha256=b.sha(v.OUT/'frozen_selection.json'),tasks=records,accepted=[r['task'] for r in records if r['accepted']],
        exact_full_v73_comparison=False,confirmation_used_for_acceptance=True,confirmation_used_for_retuning=False,test_labels_used=False)
    b.save(v.OUT/'confirmation.json',result);return result

if __name__=='__main__':
    d=b.build()
    if len(sys.argv)>1 and sys.argv[1]=='confirm':confirm(d)
    else:select(d)
