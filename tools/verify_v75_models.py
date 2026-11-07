"""Independent replay of every v75 tree/kernel and final exact CSV bytes."""
from pathlib import Path
import sys,json
import numpy as np
import joblib
from catboost import CatBoostClassifier,CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v75_experiments as v
from select_v75 import get,variants,minimum
from verify_submission import verify,read
from repack_upload_template import field_spans


def main():
    d=b.build();views=None;checks=[];sig=v.signature()
    release=json.loads((v.OUT/'release.json').read_text());frozen=json.loads((v.OUT/'frozen_selection.json').read_text());confirmed=json.loads((v.OUT/'confirmation.json').read_text())
    assert sig==release['signature']==frozen['signature']==confirmed['signature']
    assert b.sha(v.OUT/'frozen_selection.json')==confirmed['frozen_sha256']
    assert b.sha(ROOT/'tools/select_v75.py')==frozen['selection_source_sha256']
    assert b.sha(ROOT/'tools/run_v75_pipeline.py')==frozen['runner_source_sha256']
    assert sorted(confirmed['accepted'])==sorted(release['selected'])
    for k,task in enumerate(d['tasks']):
        aux_cache={}
        for file in sorted((v.OUT/task).rglob('model.cbm')):
            folder=file.parent;z=np.load(folder/'predictions.npz');state=joblib.load(folder/'state.joblib');meta=json.loads((folder/'metadata.json').read_text())
            assert str(z['signature'])==state['signature']==meta['signature']==sig;assert b.sha(file)==meta['model_sha256']
            relative=folder.relative_to(v.OUT/task);part=relative.parts[0];fit,q=v.rows(d,k,part)
            if folder.name=='inner':fit,q=v.inner(d,fit,k)
            np.testing.assert_array_equal(fit,z['fit']);np.testing.assert_array_equal(q,z['query']);np.testing.assert_array_equal(fit,state['fit']);np.testing.assert_array_equal(q,state['query']);b.assert_excluded(d,fit,q)
            conditional='conditional' in str(relative)
            aux=None
            if conditional:
                final=part=='final'
                if final not in aux_cache:aux_cache[final]=v.auxiliaries(d,k,final)
                aux,audit=aux_cache[final];assert audit==state['audit']==meta['audit'];assert task not in audit['source_tasks']
            tf=b.target(d['y'][fit],d['metrics'][k]);saved=state['target']
            for name in tf:np.testing.assert_array_equal(tf[name],saved[name])
            cls=d['metrics'][k] in ('AUROC','AUPRC');net=(CatBoostClassifier if cls else CatBoostRegressor)();net.load_model(str(file))
            x=v.tree_matrix(d,q,aux);p=v.predict_tree(net,x,d['metrics'][k],tf)
            if task=='ppbr_az':p=p.clip(0,100)
            np.testing.assert_allclose(p,z['p'],atol=1e-10,rtol=1e-10)
            probe=np.unique([len(q)-1,0,len(q)//2]);tiny=v.predict_tree(net,x[probe],d['metrics'][k],tf)
            if task=='ppbr_az':tiny=tiny.clip(0,100)
            np.testing.assert_allclose(tiny,p[probe],atol=1e-10,rtol=1e-10)
            checks.append(dict(path=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),max_error=float(np.max(abs(p-z['p']))),kind='tree'))
        for file in sorted((v.OUT/task).glob('*/kernels/predictions.npz')):
            folder=file.parent;part=folder.parent.name;fit,q=v.rows(d,k,part);z=np.load(file);assert str(z['signature'])==sig
            np.testing.assert_array_equal(fit,z['fit']);np.testing.assert_array_equal(q,z['query'])
            if views is None:views=v.fingerprints(d)
            metric=d['metrics'][k];tf=b.target(d['y'][fit],metric);meta=json.loads((folder/'metadata.json').read_text());matrices=v.kernel_matrices(d,views,fit,q)
            for j,r in enumerate(meta['models']):
                model_file=folder/r['file'];assert b.sha(model_file)==r['sha256'];state=joblib.load(model_file);assert state['signature']==sig
                np.testing.assert_array_equal(state['fit'],fit);np.testing.assert_array_equal(state['query'],q)
                for name in tf:np.testing.assert_array_equal(tf[name],state['target'][name])
                tr,x=matrices[r['family']];np.testing.assert_allclose(tr,tr.T,atol=1e-12)
                net=state['net'];p=net.decision_function(x) if metric in ('AUROC','AUPRC') else net.predict(x)
                if metric=='MAE':p=p*tf['scale']+tf['center']
                if task=='ppbr_az':p=p.clip(0,100)
                np.testing.assert_allclose(p,z['p'][j],atol=1e-10,rtol=1e-10)
                checks.append(dict(path=model_file.relative_to(ROOT).as_posix(),sha256=b.sha(model_file),max_error=float(np.max(abs(p-z['p'][j]))),kind='kernel'))
            # Reordered single-query kernel must equal the original query row.
            probe=np.unique([len(q)-1,0,len(q)//2]);small=v.kernel_matrices(d,views,fit,q[probe])
            for (tr,x),(tr2,x2) in zip(matrices,small):
                np.testing.assert_allclose(tr,tr2,atol=2e-7,rtol=2e-7);np.testing.assert_allclose(x[probe],x2,atol=2e-7,rtol=2e-7)
        b.emit('v75_verify',task=task,models=len(checks))
    cp=np.load(ROOT/'recovered_history/baseline_cv_proxy.npy');vp=np.load(ROOT/'recovered_history/baseline_valid.npy')
    for r in frozen['tasks']:
        k=d['tasks'].index(r['task']);metric=d['metrics'][k]
        for c in r['grid']:
            gs=[];pg=[]
            for part in v.P['selection_parts']:
                p,ctrl,q=get(d,k,part,c);gs.append(b.gain(metric,d['y'][q],ctrl,p));pg.append(b.gain(metric,d['y'][q],cp[q],b.blend(metric,cp[q],p,v.P['anchor_weight'])))
            np.testing.assert_allclose(gs,c['gains'],atol=1e-12);np.testing.assert_allclose(pg,c['proxy_gains'],atol=1e-12)
        ok=[c for c in r['grid'] if min(c['gains'])>0 and np.mean(c['gains'])>=minimum(metric) and np.mean(c['proxy_gains'])>0]
        expected=max(ok,key=lambda c:c['objective']) if ok else None
        assert expected==r['selection']
    for r in confirmed['tasks']:
        c=next(x for x in frozen['tasks'] if x['task']==r['task'])['selection']
        if c is None:assert not r['accepted'];continue
        assert c==r['selection'];k=d['tasks'].index(r['task']);metric=d['metrics'][k];gs=[];pg=[]
        for part in v.P['confirmation_parts']:
            p,ctrl,q=get(d,k,part,c);old=vp[q] if part=='valid' else cp[q]
            gs.append(b.gain(metric,d['y'][q],ctrl,p));pg.append(b.gain(metric,d['y'][q],old,b.blend(metric,old,p,v.P['anchor_weight'])))
        assert r['accepted']==bool(min(gs)>0 and np.mean(gs)>=minimum(metric) and min(pg)>0)
    file=ROOT/release['file'];base=next((ROOT/'admet_submissions/v73_v72').glob('*.csv'));template=ROOT/'admet_official/submission_template.csv'
    assert b.sha(file)==release['sha256'];assert b.sha(base)==v.P['anchor_sha256'];audit=verify(file,template);rows,old=read(file),read(base);changed={}
    for task in d['tasks']:
        actual=[r for r in rows[1:] if r[0]==task];prior=[r for r in old[1:] if r[0]==task]
        if task not in confirmed['accepted']:assert actual==prior;continue
        choice=release['selected'][task]['choice'];metric=release['selected'][task]['metric'];conditional=None
        if choice['conditional']:
            conditional=np.mean([np.load(v.OUT/task/'final'/f'conditional_{seed}/predictions.npz')['p'] for seed in v.P['final_seeds']],axis=0)
        p=conditional
        if choice['kernel'] is not None:
            raw=np.load(v.OUT/task/'final/kernels/predictions.npz')['p'][choice['kernel']]
            p=raw if conditional is None else b.blend(metric,raw,conditional,choice['conditional'])
        expected=b.blend(metric,np.array([float(r[4]) for r in prior]),p,v.P['anchor_weight'])
        if task=='ppbr_az':expected=expected.clip(0,100)
        np.testing.assert_array_equal(expected,np.array([float(r[4]) for r in actual]));changed[task]=sum(r[4]!=s[4] for r,s in zip(actual,prior))
    raw,ref=file.read_bytes(),template.read_bytes();s,t=field_spans(raw),field_spans(ref);chunks=[];cursor=0
    for i in range(1,len(rows)):
        lo,hi=s[i][4];a,z=t[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    result=dict(status='PASS',signature=sig,models=len(checks),final_models=sum('/final/' in r['path'] for r in checks),checks=checks,
        independent_process=True,model_reload=True,fit_query_identity=True,scaffold_exclusion=True,auxiliary_target_absent=True,
        frozen_selection_recomputed=True,confirmation_recomputed=True,fusion_recomputed=True,nonprediction_bytes_exact=True,
        changed_tasks=changed,unchanged_tasks=22-len(changed),test_labels_used=False)
    b.save(v.OUT/'model_verification.json',result);release.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification_sha256=b.sha(v.OUT/'model_verification.json'));b.save(v.OUT/'release.json',release)
    audit.update(independent_process_verification='PASS',nonprediction_bytes_exact=True,fusion_recomputed=True,changed_tasks=changed);b.save(file.with_suffix('.verification.json'),audit)
    print(json.dumps({k:x for k,x in result.items() if k!='checks'}),flush=True)


if __name__=='__main__':main()
