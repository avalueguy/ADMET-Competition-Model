"""Replay latest real v57 CSV with explicit historical-source indices, never expand CV twice."""
import csv,io,json,importlib
from pathlib import Path
import numpy as np
import admet_v24_models as b
import admet_v51_run as old
ROOT=Path(__file__).resolve().parent

def rows(path):return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))

def source(d,task,source_id,split):
    ver,index=source_id[1:].split(':');index=int(index);out=ROOT/f'admet_model_outputs_v{ver}'
    k=d['tasks'].index(task);metric=d['metrics'][k];r=np.flatnonzero((d['tid']==k)&(d['split']==(0 if split=='cv' else 1 if split=='valid' else 2)))
    if split=='cv':
        p=np.load(out/task/'cv_predictions.npy');assert p.shape[1]==len(d['y'])
        # Historical cv_predictions.npy rows are already variants.
        assert index<len(p);v=p[index,r];assert np.isfinite(v).all();return v
    z=np.load(out/task/('valid' if split=='valid' else 'final')/'predictions.npz')
    np.testing.assert_array_equal(z['query'],r);p=z['p'].astype(float)
    module=importlib.import_module(f'admet_v{ver}_run')
    if hasattr(module,'variants'):p=module.variants(p,metric)
    assert index<len(p);return p[index]

def baseline(d):
    valid,cv=old.baseline(d);f=np.load(b.OUT/'folds.npy')
    report=json.loads((ROOT/'admet_model_outputs_v49/fusion.json').read_text());replay=[]
    actual=rows(next((ROOT/'admet_submissions/v57').glob('*.csv')))
    v46=rows(next((ROOT/'admet_submissions/v46').glob('*.csv')))
    assert b.sha(next((ROOT/'admet_submissions/v57').glob('*.csv')))=='e5aef127f9d093860e2e77aad67e559bca320a34e82e8272ea066aa86354d31b'
    for task,s in report['chosen'].items():
        if task in ['lipophilicity_astrazeneca','ppbr_az']:continue
        k=d['tasks'].index(task);metric=d['metrics'][k];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        cand={}
        for mode in ['valid','cv','test']:
            a=source(d,task,s['a_id'],mode)
            if s['b_id']:
                c=source(d,task,s['b_id'],mode);a=s['alpha']*a+(1-s['alpha'])*c
            cand[mode]=a
        oldt=np.array([float(r[4]) for r in v46[1:] if r[0]==task]);newt=b.blend(metric,oldt,cand['test'],s['weight'])
        target=np.array([float(r[4]) for r in actual[1:] if r[0]==task]);err=float(np.max(np.abs(newt-target)))
        # Preserve real-file identity; don't silently assume a broken loader is equivalent.
        # v49's test writer reconstructed a v46 source incorrectly for some tasks.
        # Never replace real CSV values with that replay. Record differences;
        # these development CV estimates remain proxies, not exact full-ensemble OOF.
        valid[vi]=b.blend(metric,valid[vi],cand['valid'],s['weight'])
        for fold in range(3):
            active=f[tr]==fold;ix=tr[active];cv[ix]=b.blend(metric,cv[ix],cand['cv'][active],s['weight'])
        replay.append(dict(task=task,max_test_replay_error=err,already_expanded_cv=True,
            actual_test_from_real_v57=True,replay_exact=bool(err<2e-6)))
    fusion=json.loads((ROOT/'admet_model_outputs_v56/fusion_replay.json').read_text())
    for task,s in fusion.items():
        k=d['tasks'].index(task);tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        if s['family']=='neural':
            import admet_v52_neural as neural
            cp=np.load(neural.OUT/task/'cv_predictions.npy')[1,tr]
            vp=neural.variants(np.load(neural.OUT/task/'valid/predictions.npz')['p'])[1]
            valid[vi]=b.blend('MAE',valid[vi],vp,s['weight'])
            for fold in range(3):
                active=f[tr]==fold;cv[tr[active]]=b.blend('MAE',cv[tr[active]],cp[active],s['weight'])
        calibration=s.get('calibration')
        if calibration:
            from admet_v52_calibration import fit_lad
            p=cv[tr].copy();y=d['y'][tr];mode=calibration['nested_evaluation']['mode'];shrink=calibration['nested_evaluation']['shrink']
            for fold in range(3):
                active=f[tr]!=fold;query=~active
                beta=np.array([1.,np.median(y[active]-p[active])]) if mode=='median_bias' else fit_lad(p[active],y[active])
                beta=np.array([1.,0.])+shrink*(beta-np.array([1.,0.]))
                cv[tr[query]]=beta[0]*p[query]+beta[1]
            a,c=calibration['coefficients'];valid[vi]=a*valid[vi]+c
            if task=='ppbr_az':cv[tr]=cv[tr].clip(0,100);valid[vi]=valid[vi].clip(0,100)
    return valid,cv,replay

if __name__=='__main__':
    b.remote_only();d=b.build_data();v,c,r=baseline(d)
    dest=ROOT/'admet_model_outputs_v60';dest.mkdir(exist_ok=True)
    np.save(dest/'baseline_valid.npy',v);np.save(dest/'baseline_cv_proxy.npy',c)
    b.save(dest/'baseline_replay.json',dict(status='PASS',records=r,current_score_unknown=True,score_upper_bound=50.755035,historical_proxy=True))
    print(json.dumps(dict(status='PASS',tasks=len(r))),flush=True)
