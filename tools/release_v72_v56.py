"""Select audited endpoint changes, then preserve the exact recovered v56 base."""
from pathlib import Path
import sys, json, csv, io
from datetime import datetime, timezone, timedelta
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v71_v56_catboost as v
from analyze_v71_v56 import main as analyze
from repack_upload_template import field_spans, rows as read_bytes
from verify_submission import verify

OUT=ROOT/'admet_model_outputs_v72_v56'

def final_specialist(d,k):
    """Refit with the median development tree budget, without test labels."""
    task=d['tasks'][k];metric=d['metrics'][k]
    development=[v.OUT/task/part/'metadata.json' for part in ('fold0','fold1','fold2','valid')]
    trees=[json.loads(p.read_text())['trees'] for p in development]
    iterations=max(1,int(np.median(trees)))
    fit,q=v.rows_for(d,k,'test');b.assert_excluded(d,fit,q)
    config=dict(iterations=iterations,depth=7,learning_rate=.035,l2_leaf_reg=12,
        loss_function='Logloss' if metric in ('AUROC','AUPRC') else ('MAE' if metric=='MAE' else 'RMSE'),
        task_type='GPU',devices='0',random_seed=v.SEED,thread_count=4,verbose=False,
        allow_writing_files=False,border_count=64,bootstrap_type='Bernoulli',subsample=.85)
    signature=__import__('hashlib').sha256(json.dumps(dict(source_sha256=b.sha(__file__),
        development_signature=v.sig(),config=config,development_metadata={str(p.relative_to(ROOT)):b.sha(p) for p in development}),sort_keys=True).encode()).hexdigest()
    dest=OUT/task/'final';dest.mkdir(parents=True,exist_ok=True);cache=dest/'predictions.npz'
    state=b.preprocess(d,fit);target=b.target(d['y'][fit],metric)
    cls=metric in ('AUROC','AUPRC')
    if cache.exists():
        z=np.load(cache);assert str(z['signature'])==signature
        np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q)
        return z['p'].copy(),iterations
    x=b.transform(d['x'][d['mi'][fit]],state);qx=b.transform(d['x'][d['mi'][q]],state)
    _,inv,cnt=np.unique(d['canonical'][d['mi'][fit]],return_inverse=True,return_counts=True)
    weights=1/cnt[inv];weights/=weights.mean()
    model=(CatBoostClassifier if cls else CatBoostRegressor)(**config)
    model.fit(x,b.encode(d['y'][fit],target),sample_weight=weights)
    def predict(mod):
        p=mod.predict_proba(qx)[:,1] if cls else mod.predict(qx)
        return p*target['scale']+target['center'] if metric=='MAE' else p
    p=predict(model);model.save_model(str(dest/'model.cbm'))
    restored=(CatBoostClassifier if cls else CatBoostRegressor)();restored.load_model(str(dest/'model.cbm'))
    check=predict(restored);np.testing.assert_allclose(p,check,atol=1e-8,rtol=1e-8)
    assert np.isfinite(p).all()
    np.savez_compressed(cache,fit=fit,query=q,p=p,signature=signature)
    serial=lambda x:{key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in x.items()}
    b.save(dest/'metadata.json',dict(signature=signature,task=task,metric=metric,config=config,
        development_trees=trees,tree_budget_rule='integer median of 3 CV plus official-valid best tree counts',
        development_metadata_sha256={str(p.relative_to(ROOT)):b.sha(p) for p in development},
        state=serial(state),target_state=serial(target),fit_rows=len(fit),query_rows=len(q),
        model_sha256=b.sha(dest/'model.cbm'),prediction_sha256=b.sha(cache),
        model_reloaded=True,preprocessing_fit_only=True,test_labels_used=False))
    return p,iterations

def main():
    analyze()
    diagnostic=json.loads((OUT/'diagnostic.json').read_text())
    assert all(r['complete'] for r in diagnostic['tasks'])
    source,rows=v.load_v56();original=[r.copy() for r in rows]
    assert b.sha(source)=='ef8a199bedb658ed4fc2147581589cfd5d830a568d4505a83ff05ecfa3cfc8f3'
    d=b.build(); selected=[]
    for r in diagnostic['tasks']:
        if not r['same_v56_v57']:continue
        eligible=[w for w in r['weights'] if w['weight']<=.5 and w['valid_gain']>.0005
             and min(w['historical_proxy_cv_gains'])>0 and w['historical_proxy_cv_mean']>.001]
        if not eligible:continue
        s=max(eligible,key=lambda w:w['historical_proxy_cv_mean']-.1*np.std(w['historical_proxy_cv_gains']))
        selected.append(dict(task=r['task'],metric=r['metric'],selection=s,
             baseline_valid_score=r['proxy_valid_score'],candidate_valid_score=s['valid_score']))
    assert selected,'No endpoint with supported gain; retain v56'
    b.save(OUT/'release_selection.json',dict(status='FROZEN_BEFORE_TEST_PREDICTION',source_sha256=b.sha(source),
         selected=selected,policy='same v56/v57 CSV endpoint; positive official-valid gain >0.0005; all three historical proxy gains positive; CV mean >0.001; specialist weight <=0.5',
         score_is_platform_score=False,selection_source_sha256=b.sha(OUT/'diagnostic.json')))
    replay={}
    for r in selected:
        task=r['task'];k=d['tasks'].index(task);w=r['selection']['weight']
        p,iterations=final_specialist(d,k)
        fit,q=v.rows_for(d,k,'test');pos=[i for i,row in enumerate(rows) if i and row[0]==task]
        assert len(pos)==len(p)==len(q)
        assert [int(rows[i][1]) for i in pos]==list(range(len(p)))
        assert [rows[i][3] for i in pos]==[d['smiles'][j] for j in d['mi'][q]]
        old=np.array([float(rows[i][4]) for i in pos]);new=v.blend(r['metric'],old,p,w)
        if task=='ppbr_az':new=new.clip(0,100)
        for i,value in zip(pos,new):rows[i][4]=repr(float(value))
        replay[task]=dict(weight=w,baseline=old.tolist(),specialist=p.tolist(),result=new.tolist(),
                          changed_rows=int(np.sum(old!=new)),final_iterations=iterations)
    # Only y_pred spans change. Every other byte comes from the official template.
    template_path=ROOT/'admet_official/submission_template.csv';template=template_path.read_bytes()
    template_rows=read_bytes(template);spans=field_spans(template)
    assert len(rows)==len(template_rows)==16380
    chunks=[];cursor=0
    for i,(row,ref) in enumerate(zip(rows[1:],template_rows[1:]),1):
        assert row[:4]+row[5:]==ref[:4]+ref[5:]
        if row[0] not in replay:assert row==original[i]
        lo,hi=spans[i][4];value=row[4].encode('ascii')
        assert not any(x in value for x in b',\r\n"')
        chunks.extend((template[cursor:lo],value));cursor=hi
    chunks.append(template[cursor:]);raw=b''.join(chunks)
    actual_spans=field_spans(raw);recovered=[];cursor=0
    for i in range(1,len(rows)):
        lo,hi=actual_spans[i][4];a,c=spans[i][4]
        recovered.extend((raw[cursor:lo],template[a:c]));cursor=hi
    recovered.append(raw[cursor:]);assert b''.join(recovered)==template
    date=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')
    dest=ROOT/'admet_submissions/v72_v56';dest.mkdir(parents=True,exist_ok=True)
    file=dest/f'我从山北来带着兰花草-小分子药物预测技术创新赛-{date}.csv'
    if file.exists():assert file.read_bytes()==raw
    else:file.write_bytes(raw)
    audit=verify(file,template_path)
    audit.update(nonprediction_bytes_exact=True,unchanged_tasks_prediction_strings_exact=True,
        baseline_version='v56',baseline_official_score_user_reported=50.603,
        baseline_sha256=b.sha(source),changed_tasks={t:r['changed_rows'] for t,r in replay.items()},
        official_score=None,automatically_uploaded=False,model_reload_status='SEPARATE_CHECK_REQUIRED')
    b.save(file.with_suffix('.verification.json'),audit);b.save(OUT/'fusion_replay.json',replay)
    release=dict(status='CANDIDATE_AWAITING_INDEPENDENT_RELOAD',file=file.relative_to(ROOT).as_posix(),
        sha256=b.sha(file),template_sha256=b.sha(template_path),source_sha256=b.sha(source),
        selected=selected,official_score=None,baseline_official_score_user_reported=50.603,
        new_training_official_only=True,baseline_contains_historical_pretrained_sources=True,
        test_labels_used=False,automatic_platform_submission=False,
        limitations='Reused historical ensemble proxies; validation has been used for selection. No independent unseen-label evaluation or guaranteed score increase. Exact old ensemble weights are incomplete, so arbitrary-molecule v56 reconstruction is unavailable.')
    b.save(OUT/'release.json',release)
    print(json.dumps(dict(status='EXPORTED',file=str(file),sha256=b.sha(file),tasks=list(replay)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
