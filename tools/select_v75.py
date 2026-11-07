"""Freeze whole-method choices before confirmation, with honest proxy identity."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v75_experiments as v


def variants():
    out=[dict(name='conditional',conditional=1.,kernel=None)]
    for j in range(4):
        out.append(dict(name=f'kernel{j}',conditional=0.,kernel=j))
        out.append(dict(name=f'conditional_kernel{j}',conditional=.5,kernel=j))
    return out


def get(d,k,part,choice):
    folder=v.OUT/d['tasks'][k]/part;fit,q=v.rows(d,k,part)
    def read(path):
        z=np.load(path);assert str(z['signature'])==v.signature()
        np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q)
        return z['p']
    controls=[read(folder/'structure_tree/outer/predictions.npz')]
    for seed in v.P['final_seeds'][1:]:
        path=folder/f'structure_tree/outer_{seed}/predictions.npz'
        if path.exists():controls.append(read(path))
    control=np.mean(controls,axis=0)
    c=choice['conditional'];p=None
    if c:
        pp=[read(folder/'conditional_tree/outer/predictions.npz')]
        for seed in v.P['final_seeds'][1:]:
            path=folder/f'conditional_tree/outer_{seed}/predictions.npz'
            if path.exists():pp.append(read(path))
        p=np.mean(pp,axis=0)
    if choice['kernel'] is not None:
        raw=read(folder/'kernels/predictions.npz')[choice['kernel']]
        p=raw if not c else b.blend(d['metrics'][k],raw,p,c)
    assert p is not None
    return p,control,q


def minimum(metric):return .01 if metric=='MAE' else .003


def selection(d):
    dest=v.OUT/'frozen_selection.json'
    if dest.exists():
        result=json.loads(dest.read_text());assert result['signature']==v.signature();return result
    assert not any((v.OUT/t/part/'conditional_tree/outer/predictions.npz').exists() for t in d['tasks'] for part in v.P['confirmation_parts'])
    proxy=np.load(ROOT/'recovered_history/baseline_cv_proxy.npy');records=[]
    for k,task in enumerate(d['tasks']):
        grid=[]
        for c in variants():
            gains=[];pg=[];scores=[]
            for part in v.P['selection_parts']:
                try:p,control,q=get(d,k,part,c)
                except FileNotFoundError:break
                gains.append(b.gain(d['metrics'][k],d['y'][q],control,p))
                pg.append(b.gain(d['metrics'][k],d['y'][q],proxy[q],b.blend(d['metrics'][k],proxy[q],p,v.P['anchor_weight'])))
                scores.append(b.metric_score(d['metrics'][k],d['y'][q],p))
            if len(gains)!=2:continue
            grid.append(dict(**c,gains=gains,proxy_gains=pg,scores=scores,objective=float(np.mean(gains)-.15*np.std(gains))))
        ok=[c for c in grid if min(c['gains'])>0 and np.mean(c['gains'])>=minimum(d['metrics'][k]) and np.mean(c['proxy_gains'])>0]
        selected=max(ok,key=lambda c:c['objective']) if ok else None
        records.append(dict(task=task,metric=d['metrics'][k],selection=selected,grid=grid))
        print(json.dumps(dict(task=task,selection=selected)),flush=True)
    result=dict(status='FROZEN_BEFORE_CONFIRMATION',signature=v.signature(),tasks=records,selection_source_sha256=b.sha(__file__),
        runner_source_sha256=b.sha(ROOT/'tools/run_v75_pipeline.py'),exact_full_anchor_oof=False,anchor_weight=v.P['anchor_weight'],test_labels_used=False)
    b.save(dest,result);return result


def confirmation(d):
    frozen=json.loads((v.OUT/'frozen_selection.json').read_text());assert frozen['signature']==v.signature()
    assert frozen['selection_source_sha256']==b.sha(__file__)
    cp=np.load(ROOT/'recovered_history/baseline_cv_proxy.npy');vp=np.load(ROOT/'recovered_history/baseline_valid.npy');records=[]
    for r in frozen['tasks']:
        c=r['selection'];task=r['task'];k=d['tasks'].index(task);metric=d['metrics'][k]
        if c is None:records.append(dict(task=task,accepted=False,reason='Selection gate rejected'));continue
        parts=[]
        for part in v.P['confirmation_parts']:
            for seed in v.P['final_seeds'][1:]:
                assert (v.OUT/task/part/f'structure_tree/outer_{seed}/predictions.npz').exists()
                if c['conditional']:assert (v.OUT/task/part/f'conditional_tree/outer_{seed}/predictions.npz').exists()
            p,control,q=get(d,k,part,c);old=vp[q] if part=='valid' else cp[q]
            mixed=b.blend(metric,old,p,v.P['anchor_weight'])
            parts.append(dict(part=part,n=len(q),control_score=b.metric_score(metric,d['y'][q],control),expert_score=b.metric_score(metric,d['y'][q],p),
                gain=b.gain(metric,d['y'][q],control,p),historical_proxy_score=b.metric_score(metric,d['y'][q],old),
                proxy_blend_score=b.metric_score(metric,d['y'][q],mixed),proxy_gain=b.gain(metric,d['y'][q],old,mixed)))
        accepted=bool(min(x['gain'] for x in parts)>0 and np.mean([x['gain'] for x in parts])>=minimum(metric) and min(x['proxy_gain'] for x in parts)>0)
        records.append(dict(task=task,selection=c,parts=parts,accepted=accepted))
        print(json.dumps(records[-1]),flush=True)
    result=dict(status='CONFIRMATION_COMPLETE',signature=v.signature(),frozen_sha256=b.sha(v.OUT/'frozen_selection.json'),
        tasks=records,accepted=[r['task'] for r in records if r['accepted']],exact_full_anchor_oof=False,test_labels_used=False,confirmation_retuning=False)
    b.save(v.OUT/'confirmation.json',result);return result


if __name__=='__main__':
    d=b.build()
    confirmation(d) if len(sys.argv)>1 and sys.argv[1]=='confirm' else selection(d)
