"""Paired v24/v25 evaluation, frozen selection, final refit and CSV release."""
import argparse,csv,io,json
from pathlib import Path
import numpy as np
import pandas as pd
import admet_v24_models as b
from admet_v25_models import ROOT,OUT,P,signature,train

def v24_valid(d):
    previous=json.loads((b.OUT/'analysis.json').read_text())
    assert previous['signature']==b.signature()
    pred=np.load(b.OUT/'baseline_valid.npy').copy();candidate=np.load(b.OUT/'valid_predictions.npy')
    for row in previous['tasks']:
        if not row['weight']:continue
        k=d['tasks'].index(row['task']);ix=np.flatnonzero((d['tid']==k)&(d['split']==1))
        pred[ix]=b.blend(row['metric'],pred[ix],candidate[ix],row['weight'])
    return pred

def evaluate(d):
    folds=np.load(b.OUT/'folds.npy');cv=np.full((2,len(d['y'])),np.nan)
    for fold in range(P['folds']):
        query=np.flatnonzero(folds==fold);fit=np.flatnonzero((folds>=0)&(folds!=fold));parts=[]
        for seed in P['cv_seeds']:
            # Verify cached comparator really used identical labelled rows.
            old=np.load(b.OUT/'cv'/f'fold{fold}'/f'gated_{seed}'/'predictions.npz')
            assert str(old['signature'])==b.signature()
            np.testing.assert_array_equal(old['fit'],fit);np.testing.assert_array_equal(old['query'],query)
            parts.append(train(d,fit,query,seed,OUT/'cv'/f'fold{fold}'/str(seed)))
        cv[:,query]=np.mean(parts,0)
    np.save(OUT/'cv_predictions.npy',cv)
    query=np.flatnonzero(d['split']==1);forbidden=set(d['rg'][query])
    fit=np.flatnonzero((d['split']==0)&np.array([g not in forbidden for g in d['rg']]))
    parts=[]
    for seed in P['training_seeds']:
        old=np.load(b.OUT/'valid'/f'gated_{seed}'/'predictions.npz')
        assert str(old['signature'])==b.signature()
        np.testing.assert_array_equal(old['fit'],fit);np.testing.assert_array_equal(old['query'],query)
        parts.append(train(d,fit,query,seed,OUT/'valid'/str(seed)))
    valid=np.full((2,len(d['y'])),np.nan);valid[:,query]=np.mean(parts,0)
    np.save(OUT/'valid_predictions.npy',valid)
    seeds=np.full((len(parts),2,len(d['y'])),np.nan);seeds[:,:,query]=np.stack(parts)
    np.save(OUT/'valid_seed_predictions.npy',seeds)
    b.save(OUT/'split_audit.json',dict(folds_identical_to_v24=True,fit_query_indices_identical_to_v24=True,
           global_scaffold_overlap=0,nonisomeric_identity_overlap=0,valid_fit_rows=len(fit),valid_query_rows=len(query),
           source_folds_sha256=b.sha(b.OUT/'folds.npy'),test_labels_used=False))
    analyse(d)

def analyse(d):
    cv=np.load(OUT/'cv_predictions.npy');old=np.load(b.OUT/'cv_predictions.npy')[1]
    valid=np.load(OUT/'valid_predictions.npy');vb=v24_valid(d);np.save(OUT/'baseline_valid.npy',vb)
    f=np.load(b.OUT/'folds.npy');rows=[];nested=[]
    for k,t in enumerate(d['tasks']):
        m=d['metrics'][k];vid=np.flatnonzero((d['tid']==k)&(d['split']==1));grid=[]
        for variant,name in enumerate(P['variants']):
            for weight in P['candidate_weights']:
                gains=[];scores=[]
                for fold in range(P['folds']):
                    ix=np.flatnonzero((d['tid']==k)&(f==fold));y=d['y'][ix]
                    new=b.blend(m,old[ix],cv[variant,ix],weight)
                    gains.append(b.delta(m,y,old[ix],new))
                    scores.append(dict(fold=fold,rows=len(ix),v24_branch=b.metric(m,y,old[ix]),blend=b.metric(m,y,new)))
                grid.append(dict(variant=variant,name=name,weight=weight,fold_gains=gains,
                                 mean_cv_gain=float(np.mean(gains)),fold_scores=scores))
        # This choice cannot read the historically reused official-valid labels.
        eligible=[g for g in grid if g['mean_cv_gain']>1e-4 and sum(x>=0 for x in g['fold_gains'])>=2]
        choice=max(eligible,key=lambda g:g['mean_cv_gain']) if eligible else None
        row=dict(task=t,metric=m,grid=grid,cv_choice=choice,weight=0.,variant=None,
                 historical_valid_baseline=b.metric(m,d['y'][vid],vb[vid]),valid_rows=len(vid))
        if choice:
            variant=choice['variant'];weight=choice['weight'];new=b.blend(m,vb[vid],valid[variant,vid],weight)
            gain=b.delta(m,d['y'][vid],vb[vid],new)
            row.update(valid_blend=b.metric(m,d['y'][vid],new),valid_gain=gain,
                       candidate_valid=b.metric(m,d['y'][vid],valid[variant,vid]))
            if gain>1e-4:row.update(weight=weight,variant=variant)
            else:row['rejection']='CV choice failed historical-valid veto; no alternate search'
        else:row['rejection']='No eligible CV choice'
        # Hold a fold out of selection, choose using the other two folds only.
        held=[]
        for j in range(P['folds']):
            other=[i for i in range(P['folds']) if i!=j]
            options=[g for g in grid if np.mean([g['fold_gains'][i] for i in other])>1e-4 and all(g['fold_gains'][i]>=0 for i in other)]
            best=max(options,key=lambda g:np.mean([g['fold_gains'][i] for i in other])) if options else None
            held.append(best['fold_gains'][j] if best else 0.)
        row['held_fold_selection_gains']=held;nested.extend(held);rows.append(row)
    selected=[r for r in rows if r['weight']]
    # Fixed post-selection paired cluster bootstrap. Descriptive, not a selection-corrected CI.
    rng=np.random.default_rng(250940);ug=np.unique(d['rg'][f>=0]);gmap={g:i for i,g in enumerate(ug)}
    sets=[]
    for r in selected:
        k=d['tasks'].index(r['task'])
        for fold in range(P['folds']):
            ix=np.flatnonzero((d['tid']==k)&(f==fold));candidate=b.blend(r['metric'],old[ix],cv[r['variant'],ix],r['weight'])
            sets.append((r['metric'],d['y'][ix],old[ix],candidate,np.array([gmap[g] for g in d['rg'][ix]])))
    draws=[]
    for _ in range(250):
        mult=rng.poisson(1,len(ug));gains=[]
        for m,y,a,c,group in sets:
            ix=np.repeat(np.arange(len(y)),mult[group])
            if len(ix):gains.append(b.delta(m,y[ix],a[ix],c[ix]))
        if gains:draws.append(float(np.nanmean(gains)))
    result=dict(status='ELIGIBLE_FOR_FINAL_REFIT' if selected else 'NO_ELIGIBLE_CHANGE',signature=signature(),
           baseline_sha256=P['baseline_sha256'],baseline_user_reported_score=P['baseline_user_reported_score'],tasks=rows,
           selected_tasks=[r['task'] for r in selected],
           mean_selected_cv_gain=float(np.mean([r['cv_choice']['mean_cv_gain'] for r in selected])) if selected else None,
           selected_cv_cluster_bootstrap_95=np.quantile(draws,[.025,.975]).tolist() if draws else None,
           mean_selected_secondary_valid_gain=float(np.mean([r['valid_gain'] for r in selected])) if selected else None,
           held_fold_selection_mean_all_tasks=float(np.mean(nested)),
           interpretation='CV gains are relative MAE or absolute rank-metric gains against v24 neural branch only, NOT platform points. Full v24 ensemble valid is historically reused. Bootstrap is post-selection descriptive. No fresh independent dataset.',
           official_score_prediction=None,test_labels_used=False)
    b.save(OUT/'analysis.json',result)
    b.emit('v25_analysis',selected=result['selected_tasks'],cv_mean=result['mean_selected_cv_gain'],
           valid_mean=result['mean_selected_secondary_valid_gain'],nested_mean=result['held_fold_selection_mean_all_tasks'])

def final(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    if not report['selected_tasks']:b.emit('v25_no_candidate');return
    fit=np.flatnonzero(d['split']<2);query=np.flatnonzero(d['split']==2)
    # Adapt all tasks to preserve per-task seeded execution; train/valid labels only.
    parts=[train(d,fit,query,s,OUT/'final'/str(s)) for s in P['training_seeds']]
    p=np.full((2,len(d['y'])),np.nan);p[:,query]=np.mean(parts,0);np.save(OUT/'test_predictions.npy',p)
    compact={t:[p[v,(d['tid']==k)&(d['split']==2)].tolist() for v in range(2)] for k,t in enumerate(d['tasks'])}
    b.save(OUT/'test_predictions_compact.json',compact)
    export(d)

def export(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    source=next((ROOT/'admet_submissions/v24').glob('*.csv'));assert b.sha(source)==P['baseline_sha256']
    def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
    rows=read(source);before=[r.copy() for r in rows];template=read(ROOT/'admet_official/submission_template.csv')
    p=np.load(OUT/'test_predictions.npy');changed={};replay={}
    for r in report['tasks']:
        if not r['weight']:continue
        t=r['task'];k=d['tasks'].index(t);ix=np.flatnonzero((d['tid']==k)&(d['split']==2))
        positions=[i for i,a in enumerate(rows) if i>0 and a[0]==t]
        old=np.array([float(rows[i][4]) for i in positions]);new=b.blend(r['metric'],old,p[r['variant'],ix],r['weight'])
        for i,v in zip(positions,new):rows[i][4]=repr(float(v))
        changed[t]=int(np.sum(new!=old));replay[t]=new
    assert changed and len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,c in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==c[:4]+c[5:]
        assert np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v25';dest.mkdir(parents=True,exist_ok=True)
    name='我从山北来带着兰花草-小分子药物预测技术创新赛-20260928-v25.csv';path=dest/name
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows)
    path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'));actual=read(path)
    for t,expected in replay.items():np.testing.assert_array_equal([float(a[4]) for a in actual[1:] if a[0]==t],expected)
    for a,c in zip(actual[1:],before[1:]):
        if a[0] not in changed:assert a==c
    assert b.sha(path)!=b.sha(source)
    info=dict(status='EXPERIMENTAL_AWAITING_PLATFORM',sha256=b.sha(path),baseline_sha256=b.sha(source),baseline='v24',baseline_score=50.248759,
           rows=16379,columns=6,tasks=22,bytes=path.stat().st_size,encoding='UTF-8 BOM',line_endings='CRLF',
           changed_tasks=changed,template_identity_exact=True,model_reload_verified=True,serialized_prediction_replay=True,
           training_host=P['training_host'],local_gpu_training=False,platform_score=None,automatically_uploaded=False,
           limitation=report['interpretation'],signature=signature())
    b.save(dest/'verification.json',info);b.emit('v25_export',**info)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['evaluate','analyse','final','export'])
    args=parser.parse_args();b.remote_only();OUT.mkdir(parents=True,exist_ok=True)
    b.emit('v25_start',stage_requested=args.stage,signature=signature(),host=P['training_host'])
    d=b.build_data()
    if args.stage=='evaluate':evaluate(d)
    elif args.stage=='analyse':analyse(d)
    elif args.stage=='final':final(d)
    else:export(d)
