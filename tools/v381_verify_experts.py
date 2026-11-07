"""Replay all base/OOF/stack predictions and LP certificates without fitting."""
from pathlib import Path
import json,tarfile
import numpy as np
import joblib
import v381_oof_pareto as m
import v381_run_experts as run
ROOT=Path(__file__).resolve().parent


def main():
    d=m.load();m.setup();sig=m.signature()
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    for name,key in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256')):
        assert m.sha(ROOT/name)==manifest[key]
    for r in manifest['controls']:
        assert m.sha(ROOT/r['control'])==r['control_sha256'] and m.sha(ROOT/r['design'])==r['design_sha256']
    for row in json.loads((ROOT/'source_freeze.json').read_text())['files']:
        assert m.sha(ROOT/row['name'])==row['sha256']
    assert np.isnan(d['y'][d['split']==2]).all()
    base_records=[];oof_records=[];maximum=0.
    for path in sorted((m.OUT/'oof').glob('*/*/state.joblib')):
        t,p=path.parts[-3:-1];a=m.design(d,t,p);o=joblib.load(path)
        assert o['signature']==sig
        for key in ('fit','query'):np.testing.assert_array_equal(a[key],o[key])
        reconstructed=np.full_like(o['oof'],np.nan)
        for i,split in enumerate(o['splits']):
            tr,ho=split['train'],split['held'];sub=m.subdesign(a,tr,ho)
            assert not np.intersect1d(d['rg'][sub['fit']],d['rg'][sub['query']]).size
            reconstructed[ho]=m.replay_base(d,sub,t,p,'inner'+str(i))
        np.testing.assert_array_equal(reconstructed,o['oof'])
        np.testing.assert_array_equal(m.replay_base(d,a,t,p,'full'),o['full'])
        # Actual kernel recomputation on a small block; altering query Y cannot change it.
        small={**a,'fit':a['fit'][:32],'query':a['query'][:6],'x':a['x'][:32],'qx':a['qx'][:6]}
        changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.
        k,q=m.matrices(d,small);kk,qq=m.matrices(changed,small)
        for name in k:
            np.testing.assert_array_equal(k[name],kk[name]);np.testing.assert_array_equal(q[name],qq[name])
            assert np.linalg.eigvalsh(k[name].astype(float)).min()>-1e-4
        oof_records.append(dict(task=t,part=p,group_disjoint=True,full_oof_replay_error=0.,query_labels_excluded=True,kernels_psd_small_block=True))
        for tag in ('inner0','inner1','inner2','full'):
            bp=m.OUT/'base'/t/p/tag/'state.joblib';meta=json.loads(bp.with_name('metadata.json').read_text())
            assert m.sha(bp)==meta['model_sha256'] and meta['converged']
            base_records.append(dict(task=t,part=p,tag=tag,models=3,iterations=meta['iterations']))
    records=[]
    for path in sorted((m.OUT/'tasks').glob('*/*/*/predictions.npz')):
        t,p,name=path.parts[-4:-1];c=next(c for c in m.CONFIGS if c['name']==name)
        z=np.load(path);a=m.design(d,t,p)
        assert str(z['signature'])==sig
        for key in ('fit','query'):np.testing.assert_array_equal(z[key],a[key])
        rp=m.replay(d,t,p,c);err=float(np.max(np.abs(rp-z['p'])));assert err<1e-10;maximum=max(maximum,err)
        changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.
        np.testing.assert_array_equal(rp,m.replay(changed,t,p,c))
        state=joblib.load(path.with_name('state.joblib'));o=joblib.load(m.OUT/'oof'/t/p/'state.joblib')
        cert=m.check_certificate(o['oof'],d['y'][a['fit']],d['rg'][a['fit']],c,state)
        meta=json.loads(path.with_name('metadata.json').read_text())
        assert m.sha(path.with_name('state.joblib'))==meta['model_sha256'] and m.sha(path)==meta['prediction_sha256']
        records.append(dict(task=t,part=p,config=name,error=err,certificate=cert))
    dev=json.loads((m.OUT/'development.json').read_text());frozen=json.loads((m.OUT/'frozen_selection.json').read_text());chosen=[]
    for r in dev['records']:
        choices=[]
        for c in m.configs(r['task']):
            for w in m.WEIGHTS:
                parts=[run.score(d,r['task'],p,c,w) for p in ('fold0','fold1')];objective=min(x['gain'] for x in parts)
                if r['task'] in m.TASKS[5:] and min(x['rank_gain'] for x in parts)<=0:objective=min(objective,-1.)
                choices.append(dict(config=c,weight=w,objective=objective))
        best=max(choices,key=lambda x:(x['objective'],-x['weight'],-list(m.configs(r['task'])).index(x['config'])))
        for key in ('config','weight','objective'):assert best[key]==r['best'][key]
        if best['objective']>0:chosen.append((r['task'],best['config'],best['weight']))
    assert chosen==[(r['task'],r['config'],r['weight']) for r in frozen['selected']]
    conf=json.loads((m.OUT/'confirmation.json').read_text())
    for r in conf['records']:
        assert [run.score(d,r['task'],p,r['config'],r['weight']) for p in ('fold0','fold1','fold2','valid')]==r['confirmation']
        for b in r['bootstrap']:assert run.boot(d,r['task'],b['part'],r['config'],r['weight'])==b
    result=dict(status='PASS',signature=sig,models=len(records),base_models=sum(r['models'] for r in base_records),
                full_replay_max_error=maximum,records=records,base_records=base_records,oof_records=oof_records,no_fitting_performed=True)
    m.save(m.OUT/'independent_verification.json',result)
    # Cache matrices can be rebuilt from retained inputs; avoid transferring quadratic caches.
    files=[p for p in m.OUT.rglob('*') if p.is_file() and p.suffix!='.joblib' and '/kernels/' not in p.as_posix()]
    files += [ROOT/n for n in ('v371_mae_experts.py','v377_regression_experts.py','v380_histogram_median.py','v381_oof_pareto.py','v381_run_experts.py','v381_verify_experts.py')]
    final_models=[]
    for r in conf['records']:
        if r['experimental_eligible']:
            t=r['task']
            selected=[m.OUT/'tasks'/t/'final'/r['config']['name']/'state.joblib',m.OUT/'oof'/t/'final'/'state.joblib']
            selected += list((m.OUT/'base'/t/'final').glob('*/state.joblib'))
            for path in selected:final_models.append(dict(remote_path=str(path),sha256=m.sha(path),bytes=path.stat().st_size))
    m.save(ROOT/'result_manifest.json',dict(signature=sig,files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=m.sha(p),bytes=p.stat().st_size) for p in files],
         final_models=final_models,kernel_caches_retained_remote=True,kernel_caches_rebuildable_without_model_fitting=True))
    archive=ROOT/'results.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for path in files+[ROOT/'result_manifest.json']:tf.add(path,arcname=path.relative_to(ROOT).as_posix())
    m.save(ROOT/'result_archive.json',dict(status='PASS',signature=sig,sha256=m.sha(archive),bytes=archive.stat().st_size))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','base_records','oof_records')}))


if __name__=='__main__':main()
