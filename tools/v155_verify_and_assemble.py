"""Reload frozen development winners; verify features and export exact template."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='3'
import csv,io,json,sys
from collections import Counter
from pathlib import Path
import joblib,numpy as np,torch,xgboost as xgb
from scipy import sparse
from scipy.special import expit
from catboost import CatBoostClassifier,CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v147_path_features as features
import v148_experiment_runner as runner
import v149_target_neighbor as neighbor
import v150_path_kernels as kernel
import v151_path_boosting as boost
import v153_path_latent_trees as latent
from repack_upload_template import field_spans
from verify_submission import verify
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v155'
MODULES={'v149':neighbor,'v150':kernel,'v151':boost,'v153':latent}

def table(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))

def same(a,z):
    if sparse.issparse(a):
        assert a.shape==z.shape;assert (a!=z).nnz==0
    elif isinstance(a,dict):
        assert set(a)==set(z)
        for key in a:same(a[key],z[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(z)
        for x,y in zip(a,z):same(x,y)
    else:np.testing.assert_array_equal(a,z)

def latent_design(d,k,part):
    # Rebuild from actual saved fit-only projections, bypass the feature cache.
    fit,q,x,qx,source=v.old.features(d,k,part,'augmented');folder=latent.OUT/'features'/d['tasks'][k]/part;xx=[x];zz=[qx]
    for j,raw in enumerate(d['path_fingerprints']):
        a=raw[d['mi'][fit]].astype(float);z=raw[d['mi'][q]].astype(float);a.data=np.log1p(a.data);z.data=np.log1p(z.data)
        saved=joblib.load(folder/f'projection{j}.joblib');cols=np.flatnonzero(np.asarray((a>0).sum(0)).ravel()>=3)
        same(cols,saved['columns']);a,z=a[:,cols],z[:,cols];model=saved['model']
        # fit_transform uses U*S, transform uses X*V; differ at rounding level.
        xx.append(model.transform(a));zz.append(model.transform(z))
        freq=np.asarray((a>0).sum(0)).ravel();dense=np.argsort(freq,kind='stable')[-min(256,len(freq)):]
        same(dense,np.load(folder/f'dense_columns{j}.npy'));xx.append(a[:,dense].toarray());zz.append(z[:,dense].toarray())
    x=np.c_[*xx].astype(np.float32);qx=np.c_[*zz].astype(np.float32);x[~np.isfinite(x)]=np.nan;qx[~np.isfinite(qx)]=np.nan
    cached=latent.design(d,k,part);np.testing.assert_allclose(x,cached[2],atol=2e-5,rtol=1e-5,equal_nan=True);same(qx,cached[3])
    return fit,q,x,qx,source

def design(d,k,part,module):
    if module is kernel:return module.matrices(d,k,part)
    if module is latent:return latent_design(d,k,part)
    if module is boost:return boost.design(d,k,part)
    fit,q,x,qx,source=v.old.features(d,k,part,'augmented')
    # Always recompute neighbor summaries: a stale cache cannot prove invariance.
    tr=neighbor.summaries(d,k,fit,fit,True);te=neighbor.summaries(d,k,fit,q,False)
    cached=neighbor.design(d,k,part);same(np.c_[x,tr],cached[2]);same(np.c_[qx,te],cached[3])
    # Independently enumerate the actual permitted training-neighbor index sets.
    for row in fit:
        allowed=fit[(d['canonical'][fit]!=d['canonical'][row])&(d['groups'][fit]!=d['groups'][row])]
        assert row not in allowed and not np.any(d['canonical'][allowed]==d['canonical'][row]) and not np.any(d['groups'][allowed]==d['groups'][row])
    return fit,q,np.c_[x,tr],np.c_[qx,te],source

def reload(d,k,part,module,ch,a):
    fit,q=a[:2];cfg=ch['config'];m=d['metrics'][k];task=d['tasks'][k];folder=module.OUT/'models'/task/part;hashes={}
    if module is kernel:
        mp=folder/(module.key(cfg)+'.joblib');cache=np.load(folder/(module.key(cfg)+'.npz'));model=joblib.load(mp);qx=a[2][cfg['view']][1]
        p=expit(model.decision_function(qx)) if m in ('AUROC','AUPRC') else model.predict(qx)*float(cache['scale'])+float(cache['center']);hashes[mp.name]=b.sha(mp)
    else:
        folder=folder/module.key(cfg);cache=np.load(folder/'predictions.npz');qx=a[3];members=[]
        if module is latent:
            for seed in module.SEEDS:
                mp=folder/f'model_{seed}.cbm';model=CatBoostClassifier() if m in ('AUROC','AUPRC') else CatBoostRegressor();model.load_model(str(mp));raw=model.predict(qx,prediction_type='RawFormulaVal')
                if m=='Spearman':raw=np.mean([r83.rankdata(z)/len(z) for z in raw.T],axis=0)
                members.append(raw);hashes[mp.name]=b.sha(mp)
            p=np.mean(members,axis=0).astype(float)*float(cache['scale'])+float(cache['center'])
            if m in ('AUROC','AUPRC'):p=expit(p)
        elif module is neighbor:
            for j in range(2 if m=='Spearman' else 1):
                seeds=[]
                for seed in module.SEEDS:
                    mp=folder/f'model_t{j}_{seed}.ubj';model=xgb.Booster();model.load_model(mp);model.set_param({'device':'cpu','nthread':3});seeds.append(model.predict(xgb.DMatrix(qx,nthread=3)));hashes[mp.name]=b.sha(mp)
                raw=np.mean(seeds,axis=0).astype(float);members.append(r83.rankdata(raw)/len(raw) if m=='Spearman' else raw)
            p=np.mean(members,axis=0)*float(cache['scale'])+float(cache['center'])
        else:
            for j,cols in enumerate(a[4]):same(cols,cache[f'columns{j}'])
            for seed in module.SEEDS:
                mp=folder/f'model_{seed}.ubj';model=xgb.Booster();model.load_model(mp);model.set_param({'device':'cpu','nthread':3});members.append(model.predict(xgb.DMatrix(qx,nthread=3)));hashes[mp.name]=b.sha(mp)
            p=np.mean(members,axis=0).astype(float)*float(cache['scale'])+float(cache['center'])
    if task=='ppbr_az':p=p.clip(0,100)
    assert str(cache['signature'])==module.sig();same(cache['fit'],fit);same(cache['query'],q);np.testing.assert_allclose(p,cache['p'],atol=1e-10,rtol=0)
    return p,hashes

def perturb(d,k,part,module,before):
    q=before[1];saved=d['y'][q].copy()
    try:
        d['y'][q]=987654.;d.pop('_v96_assays',None);d.pop('_v94_aux',None);after=design(d,k,part,module);same(before,after)
    finally:
        d['y'][q]=saved;d.pop('_v96_assays',None);d.pop('_v94_aux',None)
    return True

def main():
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');OUT.mkdir(exist_ok=True)
    plan=json.loads((ROOT/'docs/v155_training_plan.json').read_text());assert plan['baseline']=='v145';experiments=[];nominees={}
    for name,module in MODULES.items():
        f=json.loads((module.OUT/'frozen_selection.json').read_text());c=json.loads((module.OUT/'confirmation.json').read_text())
        assert c['status']=='COMPLETE' and f['signature']==c['signature']==module.sig()
        experiments.append(dict(version=name,signature=module.sig(),accepted=c['accepted'],confirmation_sha256=b.sha(module.OUT/'confirmation.json')))
        for ch in f['selected']:
            conf=next(r for r in c['records'] if r['task']==ch['task']);assert all(conf[key]==value for key,value in ch.items())
            if ch['task'] not in nominees or ch['objective']>nominees[ch['task']]['record']['objective']:nominees[ch['task']]=dict(version=name,record=conf)
    selected={t:c for t,c in nominees.items() if c['record']['accepted']}
    b.save(OUT/'route_winners.json',dict(status='FROZEN_BY_DEVELOPMENT_RULE',nominees=nominees,selected=selected,experiments=experiments))
    if not selected:
        b.save(OUT/'no_candidate.json',dict(status='NO_CONFIRMED_NEW_MODEL',baseline='v145',official_score=51.166,experiments=experiments,submission_created=False))
        print('NO_CONFIRMED_NEW_MODEL',flush=True);return
    d=boost.load();original=table(v.ANCHOR);result=[r.copy() for r in original];checks=[]
    for task,ch in selected.items():
        k=d['tasks'].index(task);module=MODULES[ch['version']];r=ch['record'];pos=[i for i,z in enumerate(original) if i and z[0]==task]
        confirmed=runner.confirm(d,k,module.predict,{key:value for key,value in r.items() if key not in ('confirmation','bootstrap','accepted')});assert confirmed==r
        for part in v.PARTS:
            a=design(d,k,part,module);fit,q=a[:2];p,hashes=reload(d,k,part,module,r,a);safe=perturb(d,k,part,module,a);base=v.baseline(d,k,part);candidate=v.mix(d['metrics'][k],base,p,r['weight']);assert np.isfinite(candidate).all()
            rec=dict(task=task,version=ch['version'],part=part,fit=len(fit),query=len(q),model_reload=True,model_hashes=hashes,query_labels_recomputed_features_invariant=safe)
            if part=='final':
                assert len(pos)==len(candidate)
                for i,value in zip(pos,candidate):result[i][4]=repr(float(value))
            else:
                expected=next(z for z in r['confirmation'] if z['part']==part);np.testing.assert_allclose(r83.score(d['metrics'][k],d['y'][q],candidate),expected['candidate'],atol=1e-12,rtol=0);rec.update(expected)
            checks.append(rec);print(json.dumps(dict(stage='verified',**rec)),flush=True)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();spans=field_spans(raw);pieces=[];cursor=0
    for i,row in enumerate(result[1:],1):
        lo,hi=spans[i][4];pieces.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    pieces.append(raw[cursor:]);payload=b''.join(pieces);folder=ROOT/'admet_submissions/v155_new_structure_models';folder.mkdir(parents=True,exist_ok=True)
    file=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261027.csv'
    if file.exists():assert file.read_bytes()==payload
    else:file.write_bytes(payload)
    fmt=verify(file,template);assert fmt['rows']==16379 and fmt['tasks']==22 and fmt['utf8_bom'] and fmt['crlf_only'];changed=Counter()
    for a,z in zip(original[1:],result[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[0] not in selected:assert a==z
        if a[4]!=z[4]:changed[a[0]]+=1
    assert set(changed)==set(selected) and 'hia_hou' not in changed
    outspans=field_spans(payload);pieces=[];cursor=0
    for i in range(1,len(result)):
        lo,hi=outspans[i][4];a,z=spans[i][4];pieces.extend((payload[cursor:lo],raw[a:z]));cursor=hi
    pieces.append(payload[cursor:]);assert b''.join(pieces)==raw
    fmt.update(nonprediction_bytes_exact=True,protected_task_predictions_exact=True,changed_tasks=dict(changed),unchanged_tasks=22-len(changed))
    report=dict(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',version='v155',baseline='v145',baseline_official_score=51.166,baseline_sha256=v.ANCHOR_SHA,file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),selected=selected,experiments=experiments,checks=checks,verification=fmt,
        official_score=None,target_52_achieved=False,test_labels_used=False,external_data_added=False,automatically_uploaded=False,limitations='Public labels repeatedly reused; carried baseline controls include historical approximate OOF. Bootstrap does not remove model selection bias or guarantee an official gain.')
    b.save(OUT/'evidence.json',report);b.save(OUT/'independent_verification.json',dict(status='PASS',checks=checks,baseline=v.ANCHOR_SHA));b.save(folder/'model_evidence.json',report);b.save(file.with_suffix('.verification.json'),fmt)
    print(json.dumps(dict(status=report['status'],changed_tasks=dict(changed),sha256=report['sha256'])),flush=True)
if __name__=='__main__':main()
