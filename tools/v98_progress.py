"""Read completed development caches only; never inspect confirmation labels."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v98_shape_experts as v

d=v.load();records=[]
for k,task in enumerate(d['tasks']):
    files=[v.OUT/'models'/task/p/cfg['name']/mode/str(s)/'metadata.json' for p in ('fold0','fold1') for cfg in v.SPECS for mode in v.MODES for s in v.SEEDS]
    if not all(f.exists() for f in files):continue
    m=d['metrics'][k];bases={p:v.base(d,k,p) for p in ('fold0','fold1')};trials=[]
    for spec in v.SPECS:
      for step in v.SNAPSHOTS:
        for mode in v.MODES:
            cfg=dict(spec=spec['name'],step=step,mode=mode);ps={p:v.expert(d,k,p,cfg) for p in bases}
            for w in v.WEIGHTS:
                parts=[]
                for part,old in bases.items():
                    q=v.r83.split(d,k,part)[1];p=v.mix(m,old,ps[part],w)
                    parts.append(dict(part=part,baseline=v.r83.score(m,d['y'][q],old),candidate=v.r83.score(m,d['y'][q],p),gain=v.r83.gain(m,d['y'][q],old,p)))
                trials.append(dict(task=task,metric=m,config=cfg,weight=w,parts=parts,objective=min(x['gain'] for x in parts)))
    best=max(trials,key=lambda x:x['objective']);records.append(best)
    print(json.dumps(best),flush=True)
v.b.save(v.OUT/'development_progress.json',dict(status='PARTIAL_DEVELOPMENT_ONLY',records=records,confirmation_inspected=False))
