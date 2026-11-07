"""Independently reload the matched half-life models and quantify uncertainty."""
import json,sys
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v119_matched_half_life as v

def main():
    d=v.v.load();k=d['tasks'].index('half_life_obach')
    f=json.loads((v.OUT/'frozen_selection.json').read_text());c=json.loads((v.OUT/'confirmation.json').read_text())
    assert c['status']=='COMPLETE' and c['accepted'] and c['signature']==f['signature']==v.sig()
    assert f['record']==c['record'] and f['selected'];view=c['record']['view']
    checks=[];rng=np.random.default_rng(1204026)
    for part in ('fold0','fold1','fold2','valid','final'):
        fit,q,x,qx,source=v.design(d,k,part,view);folder=v.OUT/'models'/part/view
        z=np.load(folder/'predictions.npz');np.testing.assert_array_equal(fit,z['fit']);np.testing.assert_array_equal(q,z['query'])
        assert str(z['signature'])==v.sig();assert not np.any(d['tid'][source]==k)
        members=[];hashes={}
        for seed in v.SEEDS:
            file=folder/f'model_{seed}.cbm';net=CatBoostRegressor();net.load_model(str(file));raw=net.predict(qx)
            members.append(.5*v.r83.rankdata(raw[:,0])/len(q)+.5*v.r83.rankdata(raw[:,1])/len(q))
            hashes[file.name]=v.b.sha(file)
        expert=np.mean(members,axis=0);np.testing.assert_array_equal(expert,z['p'])
        a=v.base(d,k,part);p=.5*v.r83.rankdata(a)/len(a)+.5*v.r83.rankdata(expert)/len(expert)
        record=dict(part=part,fit=len(fit),query=len(q),model_sha256=hashes,reload_exact=True,
                    prediction_sha256=v.b.sha(folder/'predictions.npz'))
        if part!='final':
            s=v.record(d,k,part,view);expected=next(x for x in c['parts'] if x['part']==part)
            assert s==expected;record.update(s)
            if part in ('fold2','valid'):
                delta=[];groups=np.unique(d['groups'][q]);y=d['y'][q]
                # Group bootstrap is diagnostic only, not a new model selector.
                for _ in range(1000):
                    chosen=rng.choice(groups,len(groups),replace=True)
                    ix=np.concatenate([np.flatnonzero(d['groups'][q]==g) for g in chosen])
                    if len(ix)>3 and len(np.unique(y[ix]))>1:
                        value=v.r83.gain('Spearman',y[ix],a[ix],p[ix])
                        if np.isfinite(value):delta.append(value)
                record.update(scaffold_bootstrap_gain_ci95=np.quantile(delta,[.025,.975]).tolist(),
                              bootstrap_samples=len(delta),groups=len(groups))
        else:
            # Prove the baseline actually matches the task in the upload.
            import csv,io
            path=next((ROOT/'admet_submissions/v114_herg_history52').glob('*.csv'))
            rs=list(csv.DictReader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))
            old=np.array([float(r['y_pred']) for r in rs if r['task']=='half_life_obach'])
            np.testing.assert_allclose(a,old,atol=1e-14,rtol=0)
            record.update(baseline_upload_replay_exact=True,mean_rank_shift=float(np.abs(p-v.r83.rankdata(a)/len(a)).mean()),
                expert_baseline_spearman=v.r83.score('Spearman',a,expert))
        checks.append(record)
    report=dict(status='PASS',signature=v.sig(),models=2*len(checks),checks=checks,
        baseline='Exact carried v68 half-life branch, same target fit/query rows in all four comparisons.',
        limitations='Historical labels reused; bootstrap intervals may cross zero. Local confirmation cannot guarantee official improvement.')
    v.b.save(v.OUT/'independent_verification.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
