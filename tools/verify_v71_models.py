"""Independently reload CatBoost files; never trust training-time PASS flags."""
from pathlib import Path
import sys, json, platform
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
import catboost

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v71_v56_catboost as v

def main():
    d=b.build(); assert not np.isfinite(d['y'][d['split']==2]).any()
    records=[]
    for file in sorted(v.OUT.glob('*/*/predictions.npz')):
        folder=file.parent;task=folder.parent.name;k=d['tasks'].index(task);m=d['metrics'][k]
        part=folder.name
        fit,q=v.rows_for(d,k,'cv' if part.startswith('fold') else part,
                         int(part[4:]) if part.startswith('fold') else None)
        b.assert_excluded(d,fit,q)
        z=np.load(file,allow_pickle=False);assert str(z['signature'])==v.sig()
        np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q)
        meta=json.loads((folder/'metadata.json').read_text())
        assert meta['signature']==v.sig()
        rebuilt=b.preprocess(d,fit)
        state={key:np.asarray(value,dtype=rebuilt[key].dtype) if isinstance(value,list) else value for key,value in meta['state'].items()}
        for key in rebuilt:np.testing.assert_array_equal(state[key],rebuilt[key])
        qx=b.transform(d['x'][d['mi'][q]],state)
        model=(CatBoostClassifier if m in ('AUROC','AUPRC') else CatBoostRegressor)()
        model.load_model(str(folder/'model.cbm'))
        assert model.tree_count_==meta['trees']
        target=b.target(d['y'][fit],m)
        p=model.predict_proba(qx)[:,1] if m in ('AUROC','AUPRC') else model.predict(qx)
        if m=='MAE':p=p*target['scale']+target['center']
        np.testing.assert_allclose(p,z['p'],atol=1e-8,rtol=1e-8)
        assert np.isfinite(p).all()
        # Persist the target state and all hashes independently of old metadata.
        target_json={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in target.items()}
        audit=dict(status='PASS',task=task,part=part,signature=v.sig(),
             model_sha256=b.sha(folder/'model.cbm'),prediction_sha256=b.sha(file),
             metadata_sha256=b.sha(folder/'metadata.json'),fit_rows=len(fit),query_rows=len(q),
             model_reloaded=True,preprocessing_rebuilt_from_fit_only=True,
             max_absolute_prediction_error=float(np.max(np.abs(p-z['p']))),target_state=target_json,
             test_labels_used=False,
             correction='Original training metadata reload_passed flag was unverified. This separate check actually reloads the model and recomputes predictions.')
        b.save(folder/'independent_reload.json',audit)
        records.append({key:value for key,value in audit.items() if key!='target_state'})
        b.save(ROOT/'admet_model_outputs_v72_v56/model_verification.json',dict(status='IN_PROGRESS',records=records))
    final_records=[]
    for file in sorted((ROOT/'admet_model_outputs_v72_v56').glob('*/final/predictions.npz')):
        folder=file.parent;task=folder.parent.name;k=d['tasks'].index(task);m=d['metrics'][k]
        fit,q=v.rows_for(d,k,'test');b.assert_excluded(d,fit,q)
        z=np.load(file,allow_pickle=False);meta=json.loads((folder/'metadata.json').read_text())
        assert str(z['signature'])==meta['signature']
        np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q)
        rebuilt=b.preprocess(d,fit)
        state={key:np.asarray(value,dtype=rebuilt[key].dtype) if isinstance(value,list) else value for key,value in meta['state'].items()}
        for key,value in rebuilt.items():np.testing.assert_array_equal(value,state[key])
        target=b.target(d['y'][fit],m)
        for key,value in target.items():np.testing.assert_array_equal(value,meta['target_state'][key])
        assert int(np.median(meta['development_trees']))==meta['config']['iterations']
        for path,digest in meta['development_metadata_sha256'].items():assert b.sha(ROOT/path)==digest
        assert meta['model_sha256']==b.sha(folder/'model.cbm') and meta['prediction_sha256']==b.sha(file)
        # Recompute the final-training source/config signature separately.
        import hashlib
        expected=hashlib.sha256(json.dumps(dict(source_sha256=b.sha(ROOT/'tools/release_v72_v56.py'),
            development_signature=v.sig(),config=meta['config'],
            development_metadata=meta['development_metadata_sha256']),sort_keys=True).encode()).hexdigest()
        assert expected==meta['signature']
        qx=b.transform(d['x'][d['mi'][q]],state)
        model=(CatBoostClassifier if m in ('AUROC','AUPRC') else CatBoostRegressor)()
        model.load_model(str(folder/'model.cbm'));assert model.tree_count_==meta['config']['iterations']
        p=model.predict_proba(qx)[:,1] if m in ('AUROC','AUPRC') else model.predict(qx)
        if m=='MAE':p=p*target['scale']+target['center']
        np.testing.assert_allclose(p,z['p'],atol=1e-8,rtol=1e-8)
        final_records.append(dict(task=task,status='PASS',model_sha256=b.sha(folder/'model.cbm'),
            predictions_sha256=b.sha(file),signature=expected,trees=model.tree_count_,
            max_absolute_prediction_error=float(np.max(np.abs(p-z['p']))),
            preprocessing_fit_only=True,test_labels_used=False))
    assert final_records
    b.save(ROOT/'admet_model_outputs_v72_v56/model_verification.json',dict(status='PASS',records=records,
         final_records=final_records,
         python=platform.python_version(),catboost=catboost.__version__,source_signature=v.sig(),
         official_data_signature=b.signature(),test_labels_used=False))
    print(json.dumps(dict(status='PASS',models=len(records))),flush=True)

if __name__=='__main__':main()
