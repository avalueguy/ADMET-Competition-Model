"""v24 audit, group-disjoint experiments and explicit release stages."""
import argparse,csv,io,json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from admet_v24_models import (ROOT,OUT,P,save,sha,emit,remote_only,signature,build_data,folds,train_model,metric,delta,blend,rank,Preprocessor)

def baseline(d):
    info=json.loads((ROOT/'admet_submissions/v23/verification.json').read_text())
    bf=ROOT/'admet_submissions/v23/v23_upload.csv';assert sha(bf)==P['baseline_sha256']
    b23=pd.read_csv(bf,dtype=str,keep_default_na=False)
    b20=pd.read_csv(next((ROOT/'admet_submissions/v20').glob('*.csv')),dtype=str,keep_default_na=False)
    y=np.full(len(d['y']),np.nan);audit=[]
    for k,t in enumerate(d['tasks']):
        ids=np.flatnonzero((d['tid']==k)&(d['split']==1));m=d['metrics'][k]
        routefile=ROOT/'admet_model_outputs_v19'/t/'selection.json'
        active=routefile.exists() and json.loads(routefile.read_text()).get('candidate_mass',0)>0
        path=ROOT/f'admet_model_outputs_v{19 if active else 16}'/t/'validation_comparison.csv'
        c=pd.read_csv(path);np.testing.assert_array_equal(c.Y_true,d['y'][ids]);v=c.blend.to_numpy(float)
        if t=='cyp2d6_substrate_carbonmangels':
            rr=ROOT/'admet_model_outputs_v20_rare'/t
            ss=json.loads((rr/'decision.json').read_text())['selection'];cand=pd.read_csv(rr/'rare_valid.csv')
            np.testing.assert_array_equal(cand.Y_true,d['y'][ids]);v=blend(m,v,cand.Y_pred.to_numpy(float),ss['weight']) if ss['mode']=='rank' else (1-ss['weight'])*v+ss['weight']*cand.Y_pred.to_numpy(float)
        base=v.copy();alpha=info['gate'].get(t,{}).get('alpha',0.)
        if alpha:
            cand=pd.read_csv(ROOT/'admet_model_outputs_v23'/t/'valid.csv');np.testing.assert_array_equal(cand.Y_true,d['y'][ids])
            # v23 applied rank fusion to classification only; replay exactly.
            v=(1-alpha)*base+alpha*cand.Y_pred.to_numpy(float) if m not in ['AUROC','AUPRC'] else blend(m,base,cand.Y_pred.to_numpy(float),alpha)
            tv=pd.read_csv(ROOT/'admet_model_outputs_v23'/t/'test.csv').Y_pred.to_numpy(np.float32)
            old=b20.loc[b20.task.eq(t),'y_pred'].to_numpy(float)
            expected=(1-alpha)*old+alpha*tv if m not in ['AUROC','AUPRC'] else blend(m,old,tv,alpha)
            actual=b23.loc[b23.task.eq(t),'y_pred'].to_numpy(float);np.testing.assert_allclose(expected,actual,rtol=1e-7,atol=3e-6)
            v16=pd.read_csv(ROOT/'admet_release_v16'/t/'validation_predictions.csv').Y_pred.to_numpy(float)
            original_gate=(1-alpha)*v16+alpha*cand.Y_pred.to_numpy(float)
            audit.append(dict(task=t,metric=m,alpha=alpha,reported_gate_gain=delta(m,d['y'][ids],v16,original_gate),
                              exact_pipeline_replay_gain=delta(m,d['y'][ids],base,v)))
        y[ids]=v
    np.save(OUT/'baseline_valid.npy',y)
    save(OUT/'v23_audit.json',dict(baseline_sha256=sha(bf),reported_score=50.077413,issues=[
         'Train/valid/test each fit descriptor means and SDs separately',
         'Validation raw blending differed from classification rank blending at export',
         'Validation anchor was v16 while test anchor was v20',
         'No independent scaffold validation was executed in v23',
         'V23 candidate trained on train only, with no train+valid final refit'],
         changed_task_replays=audit,exact_test_prediction_replay_passed=True,
         note='Reconstructed historical valid predictions remain selection-biased; not an independent test'))
    emit('v23_audit',changed_tasks=len(audit),nonpositive_replayed_gains=sum(x['exact_pipeline_replay_gain']<=0 for x in audit))
    return y

def selftest(d):
    x=np.arange(120,dtype=np.float32).reshape(20,6);pp=Preprocessor().fit(x[:15]);z=pp.transform(x[15:]);p=pp.transform(x[[18,15,19,16,17]])
    np.testing.assert_array_equal(z[[3,0,4,1,2]],p)
    np.testing.assert_array_equal(z[:1],pp.transform(x[15:16]))
    assert np.isfinite(d['fp2']).all() and np.any(d['fp2']>0)
    assert np.isfinite(d['emb']).all() and np.any(np.std(d['emb'],0)>0)
    f=folds(d)
    for k,t in enumerate(d['tasks']):
        for fold in range(3):
            yy=d['y'][(d['tid']==k)&(f==fold)]
            assert len(yy)>0
            if d['metrics'][k] in ['AUROC','AUPRC']:assert len(np.unique(yy))==2,(t,fold)
    save(OUT/'selftest.json',dict(preprocessing_query_batch_invariance=True,nonzero_fingerprints=True,
         finite_embeddings=True,group_overlap_zero=True,all_classification_folds_have_both_classes=True))
    emit('selftest',passed=True)

def run_cv(d):
    f=folds(d);pred=np.full((2,len(d['y'])),np.nan)
    for fold in range(3):
        query=np.flatnonzero(f==fold);fit=np.flatnonzero((f>=0)&(f!=fold))
        pred[0,query]=train_model(d,fit,query,'legacy',230927,OUT/'cv'/f'fold{fold}'/'legacy')
        qq=[train_model(d,fit,query,'gated',s,OUT/'cv'/f'fold{fold}'/f'gated_{s}') for s in P['cv_seeds']]
        pred[1,query]=np.mean(qq,0)
    np.save(OUT/'cv_predictions.npy',pred);emit('cv_complete')

def valid(d):
    query=np.flatnonzero(d['split']==1)
    forbidden=set(d['rg'][query]);fit=np.flatnonzero((d['split']==0)&np.array([g not in forbidden for g in d['rg']]))
    assert not set(d['rg'][fit])&forbidden
    save(OUT/'valid_split_audit.json',dict(total_train=int(sum(d['split']==0)),retained_train=len(fit),query=len(query),global_scaffold_overlap=0,
         note='More restrictive than the historical single-task valid baseline, so direct comparison is diagnostic'))
    q=[train_model(d,fit,query,'gated',s,OUT/'valid'/f'gated_{s}') for s in P['training_seeds']]
    pred=np.full(len(d['y']),np.nan);pred[query]=np.mean(q,0);np.save(OUT/'valid_predictions.npy',pred)
    seeds=np.full((len(q),len(d['y'])),np.nan);seeds[:,query]=q;np.save(OUT/'valid_seed_predictions.npy',seeds)
    emit('valid_complete')

def analyse(d):
    b=baseline(d);v=np.load(OUT/'valid_predictions.npy');cv=np.load(OUT/'cv_predictions.npy');f=np.load(OUT/'folds.npy');rows=[]
    for k,t in enumerate(d['tasks']):
        m=d['metrics'][k];ids=np.flatnonzero((d['tid']==k)&(d['split']==1))
        gains=[];scores=[]
        for fold in range(3):
            q=np.flatnonzero((d['tid']==k)&(f==fold));gains.append(delta(m,d['y'][q],cv[0,q],cv[1,q]));scores.append(dict(fold=fold,rows=len(q),legacy=metric(m,d['y'][q],cv[0,q]),candidate=metric(m,d['y'][q],cv[1,q])))
        row=dict(task=t,metric=m,cv_gains=gains,mean_cv_gain=float(np.mean(gains)),cv_scores=scores,valid_n=len(ids),
             historical_valid_baseline=metric(m,d['y'][ids],b[ids]),candidate_valid=metric(m,d['y'][ids],v[ids]),weight=0.)
        valid_grid=[]
        for w in P['fixed_candidate_weights']:
            q=blend(m,b[ids],v[ids],w);dg=delta(m,d['y'][ids],b[ids],q)
            valid_grid.append(dict(weight=w,gain=dg,score=metric(m,d['y'][ids],q)))
        row['valid_grid']=valid_grid
        if np.mean(gains)>0 and sum(g>=0 for g in gains)>=2:
            best=max(valid_grid,key=lambda x:x['gain'])
            if best['gain']>1e-4:row.update(weight=best['weight'],secondary_valid_gain=best['gain'])
        rows.append(row)
    selected=[r for r in rows if r['weight']>0]
    # Paired cluster bootstrap of fixed candidate vs the neural comparator, all 22 tasks.
    # Fold scores are aggregated separately to avoid mixing fold-specific calibration.
    rng=np.random.default_rng(240940);draws=[]
    train=np.flatnonzero(f>=0);ug=np.unique(d['rg'][train]);gmap={g:i for i,g in enumerate(ug)}
    mapped=np.array([gmap[g] for g in d['rg'][train]]);trainpos=np.full(len(f),-1);trainpos[train]=np.arange(len(train))
    taskfold=[(k,j,np.flatnonzero((d['tid']==k)&(f==j))) for k in range(len(d['tasks'])) for j in range(3)]
    for boot in range(250):
        mult=rng.poisson(1,len(ug));gg=[]
        for k,j,q in taskfold:
            expanded=np.repeat(q,mult[mapped[trainpos[q]]])
            if len(expanded):gg.append(delta(d['metrics'][k],d['y'][expanded],cv[0,expanded],cv[1,expanded]))
        draws.append(float(np.nanmean(gg)))
    ci=np.quantile(draws,[.025,.975]).tolist()
    result=dict(status='ANALYSED_NOT_EXPORTED',signature=signature(),tasks=rows,selected_tasks=[r['task'] for r in selected],
             mean_cv_gain_all_tasks=float(np.mean([r['mean_cv_gain'] for r in rows])),cv_cluster_bootstrap_95=ci,
             mean_secondary_valid_gain_selected=float(np.mean([r['secondary_valid_gain'] for r in selected])) if selected else None,
             interpretation='Relative MAE gains and absolute rank-metric gains; NOT official points. CV comparator is the neural component only. Official valid reused historically.',
             official_score_prediction=None)
    save(OUT/'analysis.json',result);emit('analysis',selected=result['selected_tasks'],mean_cv_gain=result['mean_cv_gain_all_tasks'],cv_ci=ci)

def final(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    assert len(report['selected_tasks'])>0, 'No eligible candidate'
    fit=np.flatnonzero(d['split']<2);query=np.flatnonzero(d['split']==2)
    q=[train_model(d,fit,query,'gated',s,OUT/'final'/f'gated_{s}') for s in P['training_seeds']]
    pred=np.full(len(d['y']),np.nan);pred[query]=np.mean(q,0);np.save(OUT/'test_predictions.npy',pred)
    emit('final_training_complete')

def export(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    source=ROOT/'admet_submissions/v23/v23_upload.csv';assert sha(source)==P['baseline_sha256']
    rows=list(csv.reader(io.StringIO(source.read_bytes().decode('utf-8-sig'),newline='')))
    templ=list(csv.reader(io.StringIO((ROOT/'admet_official/submission_template.csv').read_bytes().decode('utf-8-sig'),newline='')))
    before=[r.copy() for r in rows];p=np.load(OUT/'test_predictions.npy');changed={};replay={}
    for r in report['tasks']:
        if r['weight']==0:continue
        t=r['task'];k=d['tasks'].index(t);ix=np.flatnonzero((d['tid']==k)&(d['split']==2))
        positions=[i for i,a in enumerate(rows) if i>0 and a[0]==t]
        old=np.array([float(rows[i][4]) for i in positions]);q=blend(r['metric'],old,p[ix],r['weight'])
        for i,v in zip(positions,q):rows[i][4]=repr(float(v))
        replay[t]=q.tolist();changed[t]=int(np.sum(q!=old))
    assert changed and len(rows)==len(templ)==16380 and rows[0]==templ[0]
    for a,b in zip(rows[1:],templ[1:]):
        assert len(a)==6 and a[:4]+a[5:]==b[:4]+b[5:]
        assert np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    out=ROOT/'admet_submissions/v24';out.mkdir(parents=True,exist_ok=True)
    path=out/'我从山北来带着兰花草-小分子药物预测技术创新赛-20260928-v24.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows);path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode())
    rr=list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))
    for t,expected in replay.items():np.testing.assert_array_equal([float(r[4]) for r in rr[1:] if r[0]==t],expected)
    for a,b in zip(rr[1:],before[1:]):
        if a[0] not in changed:assert a==b
    assert sha(path)!=sha(source)
    info=dict(status='EXPERIMENTAL_AWAITING_PLATFORM',baseline='v23',baseline_score=50.077413,baseline_sha256=sha(source),sha256=sha(path),rows=16379,tasks=22,columns=6,
         encoding='UTF-8 BOM',line_endings='CRLF',bytes=path.stat().st_size,changed_tasks=changed,template_identity_exact=True,
         model_reload_verified=True,serialized_prediction_replay=True,training_host=P['training_host'],local_gpu_training=False,
         cv_mean_gain=report['mean_cv_gain_all_tasks'],cv_bootstrap_95=report['cv_cluster_bootstrap_95'],
         limitation=report['interpretation'],platform_score=None,automatically_uploaded=False)
    save(out/'verification.json',info);emit('export',**info)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','evaluate','final','export']);args=a.parse_args()
    remote_only();OUT.mkdir(parents=True,exist_ok=True);d=build_data()
    if args.stage=='prepare':selftest(d);baseline(d)
    elif args.stage=='evaluate':selftest(d);baseline(d);run_cv(d);valid(d);analyse(d)
    elif args.stage=='final':final(d)
    elif args.stage=='export':export(d)
