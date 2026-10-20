"""Combine trained graph and MAE specialists with exact score-oriented replay."""
import json,csv,io,hashlib
from pathlib import Path
import numpy as np
import admet_v24_models as b
from admet_v25_run import v24_valid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v29'
P=json.loads((ROOT/'admet_v29_protocol.json').read_text())

def signature():
    from admet_v27_models import signature as s27
    from admet_v28_run import signature as s28
    from admet_v31_run import signature as s30
    return hashlib.sha256(json.dumps(dict(v27=s27(),v28=s28(),v30=s30(),files={n:b.sha(ROOT/n) for n in ['admet_v29_protocol.json','admet_v29_run.py']}),sort_keys=True).encode()).hexdigest()

def analyse(d):
    from admet_v27_models import signature as s27
    from admet_v28_run import signature as s28
    from admet_v31_run import signature as s30
    a27=json.loads((ROOT/'admet_model_outputs_v27/analysis.json').read_text());assert a27['signature']==s27()
    a28=json.loads((ROOT/'admet_model_outputs_v28/analysis.json').read_text());assert a28['signature']==s28() and a28['status']=='EVALUATED'
    a30=json.loads((ROOT/'admet_model_outputs_v31/analysis.json').read_text());assert a30['signature']==s30() and a30['status']=='EVALUATED'
    f=np.load(b.OUT/'folds.npy');old=np.load(b.OUT/'cv_predictions.npy')[1];base=v24_valid(d)
    v27=np.load(ROOT/'admet_model_outputs_v27/valid_predictions.npy');c27=np.load(ROOT/'admet_model_outputs_v27/cv_predictions.npy')
    recs=[]
    for k,t in enumerate(d['tasks']):
        vi=np.flatnonzero((d['tid']==k)&(d['split']==1));tr=np.flatnonzero((d['tid']==k)&(f>=0));m=d['metrics'][k]
        variants=[dict(name=f'graph{v}',sources=[dict(version=27,variant=v,weight=1.)],valid=v27[v,vi],cv=c27[v,tr]) for v in range(2)]
        path=ROOT/'admet_model_outputs_v28'/t/'valid/predictions.npz'
        if path.exists():
            valid28=np.load(path)['p'];cv28=np.load(ROOT/'admet_model_outputs_v28'/t/'cv_predictions.npy')
            variants.extend([dict(name=f'MAE{v}',sources=[dict(version=28,variant=v,weight=1.)],valid=valid28[v],cv=cv28[v,tr]) for v in range(2)])
        path=ROOT/'admet_model_outputs_v31'/t/'valid/predictions.npz'
        if path.exists() and (path.parent.parent/'cv_predictions.npy').exists():
            valid30=np.load(path)['p'];cv30=np.load(ROOT/'admet_model_outputs_v31'/t/'cv_predictions.npy')
            variants.extend([dict(name=f'assay{v}',sources=[dict(version=31,variant=v,weight=1.)],valid=valid30[v],cv=cv30[v,tr]) for v in range(2)])
        def assess(variant):
            grid=[]
            for weight in P['weights']:
                gains=[]
                for fold in range(3):
                    active=f[tr]==fold;ix=tr[active]
                    merged=b.blend(m,old[ix],variant['cv'][active],weight)
                    gains.append(b.delta(m,d['y'][ix],old[ix],merged))
                merged=b.blend(m,base[vi],variant['valid'],weight);vg=b.delta(m,d['y'][vi],base[vi],merged)
                objective=float(.6*vg+.4*np.mean(gains)-.1*np.std(gains))
                eligible=vg>1e-4 and np.mean(gains)>-.01 and sum(x>=0 for x in gains)>=1 and objective>1e-4
                grid.append(dict(name=variant['name'],sources=variant['sources'],weight=weight,cv_gains=gains,
                                 cv_mean=float(np.mean(gains)),valid_gain=vg,objective=objective,eligible=bool(eligible)))
            return grid
        grid=[]
        for variant in variants:grid.extend(assess(variant))
        # Evaluate one exact joint candidate when independent model families qualify.
        graphchoices=[r for r in grid if r['eligible'] and r['name'].startswith('graph')]
        maechoices=[r for r in grid if r['eligible'] and r['name'].startswith('MAE')]
        if graphchoices and maechoices:
            gg=max(graphchoices,key=lambda r:r['objective']);mm=max(maechoices,key=lambda r:r['objective'])
            gv=next(x for x in variants if x['name']==gg['name']);mv=next(x for x in variants if x['name']==mm['name'])
            joint=dict(name='graph_MAE_equal',sources=[dict(**{key:source[key] for key in ['version','variant']},weight=.5) for source in [gv['sources'][0],mv['sources'][0]]],
                       valid=(gv['valid']+mv['valid'])*.5,cv=(gv['cv']+mv['cv'])*.5)
            grid.extend(assess(joint))
        assaychoices=[r for r in grid if r['eligible'] and r['name'].startswith('assay')]
        otherchoices=[r for r in grid if r['eligible'] and not r['name'].startswith('assay')]
        if assaychoices and otherchoices:
            aa=max(assaychoices,key=lambda r:r['objective']);oo=max(otherchoices,key=lambda r:r['objective'])
            av=next(x for x in variants if x['name']==aa['name'])
            # Use primitive complementary candidate to keep replay explicit.
            oo=next((r for r in sorted(otherchoices,key=lambda r:r['objective'],reverse=True) if r['name']!='graph_MAE_equal'),None)
            if oo:
                ov=next(x for x in variants if x['name']==oo['name'])
                joint=dict(name='assay_complement_equal',sources=[dict(version=src['version'],variant=src['variant'],weight=.5) for src in [av['sources'][0],ov['sources'][0]]],
                           valid=(av['valid']+ov['valid'])*.5,cv=(av['cv']+ov['cv'])*.5)
                if m!='MAE':
                    joint['valid']=(b.rank(av['valid'])+b.rank(ov['valid']))*.5
                    for fold in range(3):
                        active=f[tr]==fold;joint['cv'][active]=(b.rank(av['cv'][active])+b.rank(ov['cv'][active]))*.5
                grid.extend(assess(joint))
        candidates=[r for r in grid if r['eligible']];selection=max(candidates,key=lambda r:r['objective']) if candidates else None
        recs.append(dict(task=t,metric=m,selection=selection,grid=grid,valid_rows=len(vi),historical_valid_baseline=b.metric(m,d['y'][vi],base[vi])))
    selected=[r for r in recs if r['selection']]
    report=dict(status='ELIGIBLE' if selected else 'NO_CHANGE',signature=signature(),tasks=recs,
                selected_tasks=[r['task'] for r in selected],baseline='v24',baseline_score=P['baseline_score'],
                mean_selected_cv_gain=float(np.mean([r['selection']['cv_mean'] for r in selected])) if selected else None,
                mean_selected_historical_valid_gain=float(np.mean([r['selection']['valid_gain'] for r in selected])) if selected else None,
                negative_cv_folds=[dict(task=r['task'],fold=j,gain=g) for r in selected for j,g in enumerate(r['selection']['cv_gains']) if g<0],
                official_score_prediction=None,limitations=P['limitations'])
    b.save(OUT/'analysis.json',report);b.emit('v29_analysis',selected=report['selected_tasks'])

def export(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    source=next((ROOT/'admet_submissions/v24').glob('*.csv'));assert b.sha(source)==P['baseline_sha256']
    def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
    rows=read(source);before=[r.copy() for r in rows];template=read(ROOT/'admet_official/submission_template.csv')
    graph=json.loads((ROOT/'admet_model_outputs_v27/test_predictions_compact.json').read_text())
    mae=json.loads((ROOT/'admet_model_outputs_v28/test_predictions_compact.json').read_text())
    assay=json.loads((ROOT/'admet_model_outputs_v31/test_predictions_compact.json').read_text())
    changed={};compact={}
    for r in report['tasks']:
        sel=r['selection']
        if sel is None:continue
        t=r['task'];pos=[i for i,a in enumerate(rows) if i>0 and a[0]==t];old=np.array([float(rows[i][4]) for i in pos])
        cand=np.zeros(len(pos))
        for src in sel['sources']:
            prediction={27:graph,28:mae,31:assay}[src['version']][t][src['variant']];assert len(prediction)==len(pos)
            if sel['name']=='assay_complement_equal' and r['metric']!='MAE':prediction=b.rank(prediction)
            cand+=src['weight']*np.array(prediction)
        merged=b.blend(r['metric'],old,cand,sel['weight'])
        for i,value in zip(pos,merged):rows[i][4]=repr(float(value))
        changed[t]=int(np.sum(merged!=old));compact[t]=cand.tolist()
    assert changed and len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,t in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==t[:4]+t[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v29';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261006.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows)
    path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'));actual=read(path)
    for t,cand in compact.items():
        sel=next(r['selection'] for r in report['tasks'] if r['task']==t)
        old=np.array([float(r[4]) for r in before[1:] if r[0]==t]);expected=b.blend(next(r['metric'] for r in report['tasks'] if r['task']==t),old,np.array(cand),sel['weight'])
        np.testing.assert_array_equal([float(r[4]) for r in actual[1:] if r[0]==t],expected)
    for a,c in zip(actual[1:],before[1:]):
        if a[0] not in changed:assert a==c
    b.save(OUT/'test_predictions_compact.json',compact)
    b.save(dest/'verification.json',dict(status='EXPERIMENTAL_AWAITING_PLATFORM',signature=signature(),sha256=b.sha(path),
       baseline='v24',baseline_sha256=b.sha(source),baseline_score=P['baseline_score'],rows=16379,tasks=22,columns=6,
       bytes=path.stat().st_size,changed_tasks=changed,encoding='UTF-8 BOM',line_endings='CRLF',template_identity_exact=True,
       prediction_fusion_replayed=True,local_gpu_training=False,training_host=P['training_host'],platform_score=None,
       automatically_uploaded=False,limitations=P['limitations']))
    b.emit('v29_export',sha256=b.sha(path),changed_tasks=changed)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['analyse','export','all']);args=parser.parse_args()
    b.remote_only();OUT.mkdir(parents=True,exist_ok=True);d=b.build_data()
    if args.stage in ['analyse','all']:analyse(d)
    if args.stage in ['export','all']:export(d)
