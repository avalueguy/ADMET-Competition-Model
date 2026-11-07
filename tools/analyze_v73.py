"""Summarize endpoint experts against honestly identified historical controls."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v73_experts as e
import admet_v73_conditional as c

def load(folder,q):
    file=folder/'predictions.npz'
    if not file.exists():return None
    z=np.load(file);np.testing.assert_array_equal(z['query'],q)
    return z['p'].copy()

def analyze(d):
    base=np.load(ROOT/'recovered_history/baseline_valid.npy')
    cp=np.load(ROOT/'recovered_history/baseline_cv_proxy.npy')
    same={r['task']:r['same_v56_v57'] for r in json.loads((ROOT/'admet_model_outputs_v72_v56/diagnostic.json').read_text())['tasks']}
    records=[]
    for k,t in enumerate(d['tasks']):
        if t in ('herg','hia_hou'):continue
        metric=d['metrics'][k];qs={part:e.rows(d,k,part)[1] for part in e.PARTS}
        pred={};available={}
        for name,root,family,seed in [('physchem',e.OUT,'physchem',e.SEED),('multiview',e.OUT,'multiview',e.SEED),('conditional',c.OUT,'aux',c.SEED),('conditional_control',c.OUT,'control',c.SEED)]:
            pp={part:load(root/t/part/family/str(seed),q) for part,q in qs.items()}
            available[name]=sum(value is not None for value in pp.values())
            if all(value is not None for value in pp.values()):pred[name]=pp
        if all(name in pred for name in ('physchem','multiview')):
            pred['structure_mean']={part:b.blend(metric,pred['physchem'][part],pred['multiview'][part],.5) for part in qs}
        if not pred:continue
        vi=qs['valid'];rec=dict(task=t,metric=metric,available=available,
            control_identity='v56/v72 supported endpoint proxy' if same[t] else 'v57 development proxy, not exact v72',
            exact_endpoint_support=same[t],proxy_valid=b.metric_score(metric,d['y'][vi],base[vi]),models={},grid=[])
        for name,pp in pred.items():
            rec['models'][name]=dict(valid=b.metric_score(metric,d['y'][vi],pp['valid']),
                cv=[b.metric_score(metric,d['y'][q],pp[part]) for part,q in qs.items() if part!='valid'])
            if name=='conditional_control':continue
            for w in (.1,.25,.5,.75,1.):
                gs=[b.gain(metric,d['y'][q],cp[q],b.blend(metric,cp[q],pp[part],w)) for part,q in qs.items() if part!='valid']
                vp=b.blend(metric,base[vi],pp['valid'],w)
                rec['grid'].append(dict(family=name,weight=w,cv_gains=gs,cv_mean=float(np.mean(gs)),
                    objective=float(np.mean(gs)-.25*np.std(gs)),valid_score=b.metric_score(metric,d['y'][vi],vp),
                    valid_gain=b.gain(metric,d['y'][vi],base[vi],vp)))
        if 'conditional' in pred and 'conditional_control' in pred:
            rec['conditional_aux_increment']={part:b.gain(metric,d['y'][q],pred['conditional_control'][part],pred['conditional'][part]) for part,q in qs.items()}
        rec['top']=sorted(rec['grid'],key=lambda r:r['objective'],reverse=True)[:5]
        records.append(rec)
    result=dict(status='DEVELOPMENT_DIAGNOSTIC',tasks=records,
        test_labels_used=False,selection_uses_repeated_development_labels=True,
        limitations='Historical CV arrays are development proxies, not independent OOF. Only marked endpoints have v56/v72 identity support. Official valid is reused for selection. Conditional models use public other-assay train labels with target value and mask absent.')
    b.save(e.OUT/'diagnostic.json',result)
    for r in records:
        eligible=[g for g in r['grid'] if g['valid_gain']>.001 and g['weight']<=.5 and sum(x>0 for x in g['cv_gains'])>=2]
        print(json.dumps(dict(task=r['task'],proxy=r['proxy_valid'],exact=r['exact_endpoint_support'],
            models=r['models'],aux=r.get('conditional_aux_increment'),best=max(eligible,key=lambda g:g['objective']) if eligible else None)),flush=True)
    return result

if __name__=='__main__':analyze(b.build())
