"""Independent saved-model, fit identity and final-fusion verification."""
from pathlib import Path
import sys,json,hashlib,platform
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v73_experts as e
import admet_v73_conditional as c
from verify_submission import verify,read
from repack_upload_template import field_spans

def check(d,v,file,conditional):
    folder=file.parent;meta=json.loads((folder/'metadata.json').read_text())
    task=meta['task'];k=d['tasks'].index(task);part=meta['part'];metric=d['metrics'][k]
    module=c if conditional else e;fit,q=module.rows(d,k,part)
    z=np.load(file,allow_pickle=False)
    assert str(z['signature'])==meta['signature']==module.signature()
    np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q)
    assert b.sha(folder/'model.cbm')==meta['model_sha256']
    assert b.sha(file)==meta.get('prediction_sha256',meta.get('predictions_sha256'))
    cls=metric in ('AUROC','AUPRC');model=(CatBoostClassifier if cls else CatBoostRegressor)()
    model.load_model(str(folder/'model.cbm'));assert model.tree_count_==meta['trees']
    if conditional:
        _,qx,audit=c.design(d,k,part,fit,q,meta['view'])
        assert audit==meta['audit'] and task not in audit['source_tasks']
        assert audit['target_aux_values_and_masks_absent'] and not audit['test_labels_used']
        tf=b.target(d['y'][fit],metric);prior=np.zeros(len(q))
    else:
        qx=e.matrix(d,v,q,meta['family']);pr=e.prior(d,v,k,fit,meta['family'])
        prior=e.prior(d,v,k,q,meta['family'])
        tf=b.target(d['y'][fit]-pr if metric=='MAE' else d['y'][fit],metric)
    for key,value in tf.items():np.testing.assert_array_equal(value,meta['target'][key])
    p=model.predict_proba(qx)[:,1] if cls else model.predict(qx)
    if metric=='MAE':p=p*tf['scale']+tf['center']+prior
    np.testing.assert_allclose(p,z['p'],atol=1e-8,rtol=1e-8)
    probe=np.unique([0,len(q)//2,len(q)-1]);pp=model.predict_proba(qx[probe])[:,1] if cls else model.predict(qx[probe])
    if metric=='MAE':pp=pp*tf['scale']+tf['center']+prior[probe]
    np.testing.assert_allclose(pp,p[probe],atol=1e-8,rtol=1e-8)
    b.assert_excluded(d,fit,q)
    return dict(path=file.relative_to(ROOT).as_posix(),task=task,part=part,
        kind='conditional' if conditional else 'structure',status='PASS',
        model_sha256=b.sha(folder/'model.cbm'),predictions_sha256=b.sha(file),
        metadata_sha256=b.sha(folder/'metadata.json'),query_rows=len(q),fit_rows=len(fit),
        max_absolute_prediction_error=float(np.max(np.abs(p-z['p']))),
        query_batch_invariance=True,fit_query_identity_verified=True,test_labels_used=False)

def main():
    d=b.build();v=e.features(d);assert not np.isfinite(d['y'][d['split']==2]).any()
    release=json.loads((e.OUT/'release.json').read_text());records=[]
    # Check all development models that affected selection and all final weights.
    for conditional,module in ((False,e),(True,c)):
        for file in sorted(module.OUT.glob('*/*/*/*/predictions.npz')):
            records.append(check(d,v,file,conditional))
            if len(records)%25==0:b.emit('v73_reload',checked=len(records))
    candidate=ROOT/release['file'];baseline=ROOT/'admet_submissions/v72_v56/我从山北来带着兰花草-小分子药物预测技术创新赛-20261021.csv'
    template=ROOT/'admet_official/submission_template.csv';audit=verify(candidate,template)
    assert b.sha(candidate)==release['sha256'] and b.sha(baseline)=='9ac57cda636ea4cd7e8909e97ecde4ec656e51f024a73cd0fb6eaa29dd55f8b2'
    new,old=read(candidate),read(baseline);changed={}
    for k,task in enumerate(d['tasks']):
        a=[r for r in new[1:] if r[0]==task];z=[r for r in old[1:] if r[0]==task]
        if task not in release['selected']:
            assert a==z;continue
        rec=release['selected'][task];conditional=rec['kind']=='conditional';module=c if conditional else e
        family=rec.get('family',rec.get('view'));seed0=c.SEED if conditional else e.SEED
        trees=[json.loads((module.OUT/task/part/family/str(seed0)/'metadata.json').read_text())['trees'] for part in ('fold0','fold1','fold2','valid')]
        assert int(np.median(trees))==rec['iterations'] and trees==rec['development_trees']
        ps=[]
        for seed in (732026,732027):
            folder=module.OUT/task/'final'/family/str(seed)
            meta=json.loads((folder/'metadata.json').read_text());assert meta['config']['iterations']==rec['iterations']
            ps.append(np.load(folder/'predictions.npz')['p'])
        prediction=np.mean(ps,axis=0);oldp=np.array([float(r[4]) for r in z]);expected=b.blend(d['metrics'][k],oldp,prediction,rec['weight'])
        np.testing.assert_array_equal(expected,np.array([float(r[4]) for r in a]))
        r=release['replay'][task]
        assert hashlib.sha256(prediction.tobytes()).hexdigest()==r['prediction_sha256']
        assert hashlib.sha256(expected.tobytes()).hexdigest()==r['result_sha256']
        changed[task]=sum(x[4]!=y[4] for x,y in zip(a,z))
    raw=candidate.read_bytes();ref=template.read_bytes();ss=field_spans(raw);ts=field_spans(ref);chunks=[];cursor=0
    for i in range(1,len(new)):
        lo,hi=ss[i][4];a,z=ts[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    final=[r for r in records if r['part']=='final'];assert len(final)==10
    result=dict(status='PASS',models=len(records),final_models=len(final),records=records,
        independent_process=True,model_reloaded=True,query_batch_invariance=True,
        fusion_replay='PASS',nonprediction_bytes_exact=True,unchanged_tasks=22-len(changed),
        changed_tasks=changed,test_labels_used=False,python=platform.python_version(),
        source_sha256=b.sha(__file__),official_data_signature=b.signature())
    b.save(e.OUT/'model_verification.json',result)
    audit.update(independent_model_reload='PASS',fusion_replay='PASS',nonprediction_bytes_exact=True,unchanged_tasks=22-len(changed),changed_tasks=changed)
    b.save(candidate.with_suffix('.verification.json'),audit)
    release.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',independent_verification_sha256=b.sha(e.OUT/'model_verification.json'),baseline_contains_historical_pretrained_sources=True,new_experts_pretrained_weights=False)
    b.save(e.OUT/'release.json',release)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)

if __name__=='__main__':main()
