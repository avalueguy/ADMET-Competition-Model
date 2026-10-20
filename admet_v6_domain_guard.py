"""Post-selection safety overlay for observed low-similarity subgroup regressions.

This is a disclosed additional reused-validation check, not an independent test.
The rule is locked: fallback to baseline below Morgan similarity 0.3, and retain
the overlay only if it also passes the original observed primary/secondary gates.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from admet_v6_runner import load_task, metrics, kind, deltas, secondary_ok, save


def nearest(raw, fit, query):
    a=raw[fit,:2048];norm=a.sum(1);out=[]
    for start in range(0,len(query),256):
        b=raw[query[start:start+256],:2048];dot=b@a.T
        out.extend((dot/np.maximum(b.sum(1)[:,None]+norm[None,:]-dot,1)).max(1))
    return np.asarray(out)


def mixed(base,candidate,alpha,similarity):
    return base+np.where(similarity<.3,0.,alpha)*(candidate-base)


def get_oof(task,family):
    if '+' in family:return np.mean([get_oof(task,f) for f in family.split('+')],axis=0)
    folder='admet_model_outputs_v5' if family.startswith('v5_') else 'admet_model_outputs_v6'
    name=family[3:] if family.startswith('v5_') else family
    return pd.read_csv(Path(folder)/task/(name+'_oof.csv')).Y_pred.to_numpy(float)


def main():
    uncertainty=pd.read_csv('admet_diagnostics/v6_uncertainty.csv');records=[]
    for _,r in uncertainty.iterrows():
        if not np.isfinite(r.get('distant_primary_before',np.nan)):continue
        gain=r.distant_primary_after-r.distant_primary_before
        if r.metric=='MAE':gain/=max(abs(r.distant_primary_before),1e-8)
        tolerance=.03 if r.metric=='MAE' else .01
        if gain>=-tolerance:continue
        task=r.task;dest=Path('admet_model_outputs_v6')/task;path=dest/'decision.json'
        d=json.loads(path.read_text());save(dest/'decision_before_domain_guard.json',d)
        if not d['accepted']:continue
        data=load_task(task);n=len(data['frames']['train']);nv=len(data['frames']['valid'])
        sel=d['selection'];alpha=sel['alpha'];metric=sel['metric'];kk=kind(metric)
        checks=[];b=pd.read_csv(dest/'baseline_oof.csv');cand=get_oof(task,sel['family'])
        fold=b.fold.to_numpy();y=b.Y_true.to_numpy();pbase=b.Y_pred.to_numpy()
        gains=[]
        for f in range(3):
            it=np.flatnonzero(fold!=f);iv=np.flatnonzero(fold==f)
            q=mixed(pbase[iv],cand[iv],alpha,nearest(data['raw'],it,iv))
            old=metrics(kk,y[iv],pbase[iv]);new=metrics(kk,y[iv],q)
            change=deltas(metric,old,new);gains.append(change)
            checks.append(dict(split=f'fold_{f}',before=old,after=new,passed=bool(change>=(-.01 if metric=='MAE' else -.005) and secondary_ok(metric,old,new))))
        for name,file in [('valid','validation_comparison.csv'),('repeat','repeat_confirmation.csv')]:
            frame=pd.read_csv(dest/file);yy=frame.Y_true.to_numpy();base=frame.baseline.to_numpy();c=frame.candidate.to_numpy()
            iv=np.arange(n,n+nv) if name=='valid' else frame.row.to_numpy(int)
            it=np.arange(n) if name=='valid' else np.setdiff1d(np.arange(n),iv)
            q=mixed(base,c,alpha,nearest(data['raw'],it,iv));old=metrics(kk,yy,base);new=metrics(kk,yy,q)
            change=deltas(metric,old,new)
            check=change>= (0. if name=='valid' else (-.01 if metric=='MAE' else -.005)) and secondary_ok(metric,old,new)
            checks.append(dict(split=name,before=old,after=new,passed=bool(check)))
            frame['blend_guarded']=q;frame.to_csv(dest/(name+'_domain_guard.csv'),index=False)
        ok=all(c['passed'] for c in checks) and np.mean(gains)>0
        d.update(accepted=bool(ok),domain_guard=dict(threshold=.3,trigger_subgroup_delta=float(gain),
                     checks=checks,passed=bool(ok),limitation='Additional post-selection check on reused data'),
                 reason='domain fallback passed observed gates' if ok else 'distant subgroup risk: safety fallback failed gates')
        if ok:
            d['before']=checks[-2]['before'];d['after']=checks[-2]['after']
            guarded=pd.read_csv(dest/'valid_domain_guard.csv');guarded['blend']=guarded.blend_guarded
            original=dest/'validation_comparison_before_guard.csv'
            original.write_bytes((dest/'validation_comparison.csv').read_bytes())
            guarded.to_csv(dest/'validation_comparison.csv',index=False)
        save(path,d);records.append(dict(task=task,adopted=bool(ok),checks=checks))
    save('admet_diagnostics/v6_domain_guard.json',records);print(json.dumps(records),flush=True)


if __name__=='__main__':main()
