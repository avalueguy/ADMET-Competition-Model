"""Independent cumulative reconstruction from v7 to the scored guard, then v13."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from admet_v6_runner import remote_only, save, sha, get_metric
from admet_v10_cumulative_audit import evaluate_split, read_csv, read_json, SOURCES
from admet_v13_models import load_task
from admet_v13_runner import repeat_baseline


def main():
    remote_only()
    root=Path('admet_release_v13');prior=Path('admet_release_v11')
    v=read_json(root/'independent_verification.json');assert v['status']=='VERIFIED'
    anchor=next(Path('admet_submissions/v11').glob('*.csv'));final=Path(v['submission'])
    assert sha(anchor)=='082183f0c462081ea8735a41ec2e47326ef4cdcaf9d1188ae86a006f8057122d'
    assert sha(final)==v['sha256']
    a=pd.read_csv(anchor,dtype=str,keep_default_na=False);z=pd.read_csv(final,dtype=str,keep_default_na=False)
    assert a.drop(columns='y_pred').equals(z.drop(columns='y_pred'))
    changed=set(z.loc[z.y_pred!=a.y_pred,'task']);assert changed==set(v['accepted'])
    checks=[];chains=[]
    for task in sorted(changed):
        d=load_task(task);metric=get_metric(task);n=len(d['frames']['train']);nv=len(d['frames']['valid'])
        b=read_csv(f'admet_model_outputs_v8/{task}/baseline_oof.csv')
        cumulative=b.Y_pred.to_numpy().copy();chain=[]
        if task!='dili':
            for version in [8,9,10]:
                r=read_json(f'admet_release_v{version}/{task}/selection.json')
                if r['accepted']:
                    s=r['selection'];c=read_csv(f'admet_model_outputs_v{version}/{task}/{s["family"]}_oof.csv')
                    np.testing.assert_array_equal(c.row,b.row);np.testing.assert_array_equal(c.fold,b.fold)
                    np.testing.assert_allclose(c.Y_true,b.Y_true)
                    cumulative=(1-s['alpha'])*cumulative+s['alpha']*c.Y_pred.to_numpy();chain.append(version)
        r11=read_json(f'admet_release_v11/{task}/selection.json')
        if r11['accepted']:
            s11=r11['selection'];c11=read_csv(f'admet_model_outputs_v11/{task}/{s11['family']}_oof.csv')
            np.testing.assert_array_equal(c11.fold,b.fold);np.testing.assert_allclose(c11.Y_true,b.Y_true)
            cumulative=(1-s11['alpha'])*cumulative+s11['alpha']*c11.Y_pred.to_numpy();chain.append(11)
        recorded=read_csv(f'admet_model_outputs_v13/{task}/baseline_oof.csv')
        np.testing.assert_allclose(recorded.Y_pred,cumulative,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(b.Y_true,d['frames']['train'].Y)
        route=read_json(root/task/'selection.json');s=route['selection']
        c=read_csv(f'admet_model_outputs_v13/{task}/{s["family"]}_oof.csv')
        np.testing.assert_array_equal(c.row,b.row);np.testing.assert_array_equal(c.fold,b.fold)
        predicted=(1-s['alpha'])*cumulative+s['alpha']*c.Y_pred.to_numpy()
        for fold in range(3):
            it=np.flatnonzero(b.fold!=fold);iv=np.flatnonzero(b.fold==fold)
            checks.extend(evaluate_split(task,f'fold_{fold}',d,it,iv,b.Y_true.to_numpy()[iv],cumulative[iv],predicted[iv],metric))
        valid_base=read_csv(prior/task/'validation_predictions.csv');valid_new=read_csv(root/task/'validation_predictions.csv')
        source=read_csv(f'admet_release_v11/{task}/validation_predictions.csv')
        np.testing.assert_array_equal(valid_base.Y_pred,source.Y_pred)
        np.testing.assert_allclose(valid_new.Y_true,valid_base.Y_true)
        checks.extend(evaluate_split(task,'valid',d,np.arange(n),np.arange(n,n+nv),valid_base.Y_true.to_numpy(),valid_base.Y_pred.to_numpy(),valid_new.Y_pred.to_numpy(),metric))
        it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=692509).split(np.arange(n),groups=d['groups'][:n]))
        rep=read_csv(f'admet_model_outputs_v13/{task}/repeat_comparison.csv');np.testing.assert_array_equal(rep.row,iv)
        base=repeat_baseline(task,d,it,iv);np.testing.assert_allclose(rep.baseline,base,rtol=1e-12,atol=1e-12)
        checks.extend(evaluate_split(task,'repeat',d,it,iv,rep.Y_true.to_numpy(),base,rep.blend.to_numpy(),metric))
        chains.append(dict(task=task,baseline_adopted_versions=chain,dili_restored_to_v7=task=='dili',v13_family=s['family'],v13_alpha=s['alpha']))
    failures=[x for x in checks if x['primary_pass'] is False or x['secondary_pass'] is False]
    insufficient=[x for x in checks if not x['coverage_sufficient']]
    result=dict(status='PASSED' if not failures else 'FAILED',anchor_score=49.381036,anchor_release='v11',
        anchor_sha256=sha(anchor),final_sha256=sha(final),changed_tasks=sorted(changed),checks=len(checks),
        violations=failures,insufficient_coverage=insufficient,chains=chains,sources=SOURCES,
        caveat='Reused data. Passing fixed tolerances is not a guarantee about hidden-set performance.')
    save('admet_diagnostics/v13_anchor_audit.json',result)
    pd.DataFrame(checks).to_csv('admet_diagnostics/v13_anchor_audit.csv',index=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','chains']}),flush=True)
    assert not failures and not insufficient


if __name__=='__main__':main()
