"""Frozen development selection and strict confirmation for new task models."""
import hashlib,json
import numpy as np
from scipy.special import ndtr
import v314_current_baseline as v
b,r83=v.b,v.r83
WEIGHTS=(.2,.45)

def record(d,k,part,predict,c,w):
    q=v.split(d,k,part)[1];m=d['metrics'][k];a=v.baseline(d,k,part);z=predict(d,k,part,c);p=v.mix(m,a,z,w)
    return dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],a),expert=r83.score(m,d['y'][q],z),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],a,p))

def confirm(d,k,predict,ch):
    parts=[record(d,k,p,predict,ch['config'],ch['weight']) for p in ('fold0','fold1','fold2','valid')]
    m=d['metrics'][k];statistics=[]
    # Paired scaffold group bootstrap on frozen predictions. No choices depend on CIs.
    for part in ('fold2','valid'):
        q=v.split(d,k,part)[1];a=v.baseline(d,k,part);z=predict(d,k,part,ch['config']);p=v.mix(m,a,z,ch['weight']);groups=np.unique(d['groups'][q]);ixs={g:np.flatnonzero(d['groups'][q]==g) for g in groups}
        rng=np.random.default_rng(1484026+k);delta=[]
        if len(groups)>1:
            for _ in range(500):
                ix=np.concatenate([ixs[g] for g in rng.choice(groups,len(groups),replace=True)])
                if m in ('AUROC','AUPRC') and len(np.unique(d['y'][q][ix]))<2:continue
                delta.append(r83.gain(m,d['y'][q][ix],a[ix],p[ix]))
        lower=float(np.quantile(delta,.05)) if delta else None
        statistics.append(dict(part=part,groups=len(groups),gain_ci90=np.quantile(delta,[.05,.95]).tolist() if delta else None,lower_positive=lower is not None and lower>0))
    # Require point gains in every part and confidence support in both confirmation parts.
    accepted=bool(min(r['gain'] for r in parts)>0 and np.mean([r['gain'] for r in parts])>v.threshold(m) and all(s['lower_positive'] for s in statistics))
    return dict(**ch,confirmation=parts,bootstrap=statistics,accepted=accepted)

def run(module,d):
    out=module.OUT;out.mkdir(exist_ok=True);tasks=v.tasks(d);dev=[];selected=[]
    doc=dict(signature=module.sig(),tasks=tasks,configs=module.CONFIGS,weights=WEIGHTS,baseline='v313',method=module.METHOD,
        selection='Max minimum fold0/1 gain, single config and blend frozen before fold2/valid; no fallback.',confirmation='All4 point gains>0, mean practical threshold, both paired scaffold-bootstrap90 lower bounds>0. Single-group confirmation is rejected.',
        test_labels_used=False,external_data=False,limitations='Public labels reused. Current carried branch reconstructed but historical upstream OOF remains approximate. No official score estimate.')
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(doc))
    else:b.save(path,doc)
    for task in tasks:
        k=d['tasks'].index(task);choices=[]
        for c in module.CONFIGS:
            for w in WEIGHTS:
                parts=[record(d,k,p,module.predict,c,w) for p in ('fold0','fold1')];choices.append(dict(task=task,metric=d['metrics'][k],config=c,weight=w,parts=parts,objective=min(r['gain'] for r in parts)))
        best=max(choices,key=lambda r:r['objective']);dev.extend(choices)
        if best['objective']>v.threshold(d['metrics'][k]):selected.append(best)
        b.save(out/'development.json',dict(status='IN_PROGRESS',signature=module.sig(),records=dev));print(json.dumps(dict(stage='development',best=best)),flush=True)
    b.save(out/'development.json',dict(status='COMPLETE',signature=module.sig(),records=dev));b.save(out/'frozen_selection.json',dict(signature=module.sig(),selected=selected))
    records=[]
    for ch in selected:
        k=d['tasks'].index(ch['task']);r=confirm(d,k,module.predict,ch);records.append(r)
        if r['accepted']:module.predict(d,k,'final',ch['config'])
        b.save(out/'confirmation.json',dict(status='IN_PROGRESS',signature=module.sig(),records=records));print(json.dumps(dict(stage='confirmation',**r)),flush=True)
    b.save(out/'confirmation.json',dict(status='COMPLETE',signature=module.sig(),records=records,accepted=[r['task'] for r in records if r['accepted']]))
