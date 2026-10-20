"""Versioned graph evaluation, score-oriented selection and audited export."""
import argparse,csv,hashlib,io,json
from pathlib import Path
import numpy as np
import admet_v24_models as b
from admet_v25_run import v24_valid
from admet_v27_models import ROOT,OUT,P,signature,graphs,selftest,fit

def valid(d,g):
    query=np.flatnonzero(d['split']==1);forbidden=set(d['rg'][query])
    rows=np.flatnonzero((d['split']==0)&np.array([v not in forbidden for v in d['rg']]))
    parts=[fit(d,g,rows,query,seed,OUT/'valid'/str(seed)) for seed in P['training_seeds']]
    pred=np.full((2,len(d['y'])),np.nan);pred[:,query]=np.mean(parts,0)
    np.save(OUT/'valid_predictions.npy',pred)
    b.save(OUT/'valid_split_audit.json',dict(fit_rows=len(rows),query_rows=len(query),global_group_overlap=0,
        same_groups_as_v24=True,test_labels_used=False))
    baseline=v24_valid(d);np.save(OUT/'baseline_valid.npy',baseline)
    scores=[]
    for k,t in enumerate(d['tasks']):
        ix=query[d['tid'][query]==k];m=d['metrics'][k];base=b.metric(m,d['y'][ix],baseline[ix])
        scores.append(dict(task=t,metric=m,baseline=base,candidates=[b.metric(m,d['y'][ix],pred[v,ix]) for v in range(2)],
           best_blend_gain=max(b.delta(m,d['y'][ix],baseline[ix],b.blend(m,baseline[ix],pred[v,ix],w)) for v in range(2) for w in P['selection_weights'])))
    b.save(OUT/'valid_diagnostic.json',dict(tasks=scores,limitation='Historically reused official valid; no CV selection yet'))
    b.emit('v27_valid_complete',positive_blends=sum(r['best_blend_gain']>.0001 for r in scores))

def cv(d,g):
    folds=np.load(b.OUT/'folds.npy');pred=np.full((2,len(d['y'])),np.nan)
    for fold in range(3):
        query=np.flatnonzero(folds==fold);rows=np.flatnonzero((folds>=0)&(folds!=fold));parts=[]
        for seed in P['training_seeds']:
            parts.append(fit(d,g,rows,query,seed,OUT/'cv'/f'fold{fold}'/str(seed)))
        pred[:,query]=np.mean(parts,0)
    np.save(OUT/'cv_predictions.npy',pred);b.emit('v27_cv_complete')

def analyse(d):
    cvpred=np.load(OUT/'cv_predictions.npy');vpred=np.load(OUT/'valid_predictions.npy')
    baseline=np.load(OUT/'baseline_valid.npy');folds=np.load(b.OUT/'folds.npy');old=np.load(b.OUT/'cv_predictions.npy')[1]
    records=[]
    for k,t in enumerate(d['tasks']):
        metric=d['metrics'][k];vi=np.flatnonzero((d['tid']==k)&(d['split']==1));grid=[]
        for variant,name in enumerate(P['variants']):
            for weight in P['selection_weights']:
                if weight==0:continue
                fg=[]
                for fold in range(3):
                    ix=np.flatnonzero((d['tid']==k)&(folds==fold))
                    fg.append(b.delta(metric,d['y'][ix],old[ix],b.blend(metric,old[ix],cvpred[variant,ix],weight)))
                vg=b.delta(metric,d['y'][vi],baseline[vi],b.blend(metric,baseline[vi],vpred[variant,vi],weight))
                objective=.6*vg+.4*np.mean(fg)-.1*np.std(fg)
                eligible=vg>1e-4 and np.mean(fg)>-.01 and sum(v>=0 for v in fg)>=1 and objective>1e-4
                grid.append(dict(variant=variant,name=name,weight=weight,cv_gains=fg,cv_mean=float(np.mean(fg)),
                                  valid_gain=vg,objective=float(objective),eligible=bool(eligible)))
        eligible=[r for r in grid if r['eligible']];chosen=max(eligible,key=lambda r:r['objective']) if eligible else None
        records.append(dict(task=t,metric=metric,selection=chosen,grid=grid,valid_rows=len(vi),
           historical_valid_baseline=b.metric(metric,d['y'][vi],baseline[vi])))
    selected=[r for r in records if r['selection']]
    b.save(OUT/'analysis.json',dict(status='ELIGIBLE_FINAL' if selected else 'NO_ELIGIBLE_CHANGE',signature=signature(),
        baseline='v24',baseline_score=P['baseline_score_user_reported'],tasks=records,selected_tasks=[r['task'] for r in selected],
        mean_selected_cv_gain=float(np.mean([r['selection']['cv_mean'] for r in selected])) if selected else None,
        mean_selected_valid_gain=float(np.mean([r['selection']['valid_gain'] for r in selected])) if selected else None,
        cv_negative_folds=[dict(task=r['task'],fold=j,gain=v) for r in selected for j,v in enumerate(r['selection']['cv_gains']) if v<0],
        official_score_prediction=None,limitation=P['limitation']))
    b.emit('v27_analysis',selected=[r['task'] for r in selected])

def final(d,g):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    if not report['selected_tasks']:b.emit('v27_no_export');return
    rows=np.flatnonzero(d['split']<2);query=np.flatnonzero(d['split']==2)
    parts=[fit(d,g,rows,query,seed,OUT/'final'/str(seed)) for seed in P['final_seeds']]
    pred=np.full((2,len(d['y'])),np.nan);pred[:,query]=np.mean(parts,0);np.save(OUT/'test_predictions.npy',pred)
    compact={t:[pred[v,(d['tid']==k)&(d['split']==2)].tolist() for v in range(2)] for k,t in enumerate(d['tasks'])}
    b.save(OUT/'test_predictions_compact.json',compact);export(d)

def export(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    source=next((ROOT/'admet_submissions/v24').glob('*.csv'));assert b.sha(source)==P['baseline_sha256']
    def read(path):return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))
    rows=read(source);before=[r.copy() for r in rows];template=read(ROOT/'admet_official/submission_template.csv')
    pred=np.load(OUT/'test_predictions.npy');changed={};expected={}
    for r in report['tasks']:
        if not r['selection']:continue
        sel=r['selection'];t=r['task'];k=d['tasks'].index(t);ix=np.flatnonzero((d['tid']==k)&(d['split']==2))
        positions=[i for i,a in enumerate(rows) if i>0 and a[0]==t];base=np.array([float(rows[i][4]) for i in positions])
        new=b.blend(r['metric'],base,pred[sel['variant'],ix],sel['weight'])
        for i,p in zip(positions,new):rows[i][4]=repr(float(p))
        changed[t]=int(np.sum(base!=new));expected[t]=new
    assert changed and len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,c in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==c[:4]+c[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v27';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261006-v27.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows)
    path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'));actual=read(path)
    for t,new in expected.items():np.testing.assert_array_equal([float(a[4]) for a in actual[1:] if a[0]==t],new)
    for a,c in zip(actual[1:],before[1:]):
        if a[0] not in changed:assert a==c
    b.save(dest/'verification.json',dict(status='EXPERIMENTAL_AWAITING_PLATFORM',signature=signature(),sha256=b.sha(path),
       baseline_sha256=b.sha(source),baseline='v24',baseline_score=P['baseline_score_user_reported'],rows=16379,columns=6,tasks=22,
       bytes=path.stat().st_size,changed_tasks=changed,encoding='UTF-8 BOM',line_endings='CRLF',
       template_identity_exact=True,model_reload_verified=True,prediction_fusion_replayed=True,
       local_gpu_training=False,training_host=P['training_host'],platform_score=None,automatically_uploaded=False,
       limitation=P['limitation']))
    b.emit('v27_export',sha256=b.sha(path),changed_tasks=changed)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','valid','cv','analyse','final','all'])
    args=parser.parse_args();b.remote_only();OUT.mkdir(parents=True,exist_ok=True)
    b.emit('v27_start',stage_requested=args.stage,signature=signature());d=b.build_data();g=graphs(d)
    if args.stage=='prepare':selftest(d,g)
    elif args.stage=='valid':valid(d,g)
    elif args.stage=='cv':cv(d,g)
    elif args.stage=='analyse':analyse(d)
    elif args.stage=='final':final(d,g)
    else:selftest(d,g);valid(d,g);cv(d,g);analyse(d);final(d,g)
