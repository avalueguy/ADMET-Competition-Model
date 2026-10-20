"""Posthoc low-weight v25-model mixture; remote train, audit and export."""
import csv,hashlib,io,json,os
from pathlib import Path
import numpy as np
import admet_v24_models as b
import admet_v25_run as v25
from admet_v25_models import train,signature as model_signature

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_model_outputs_v26'
P=json.loads((ROOT/'admet_v26_protocol.json').read_text())

def read(path):return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))
def runsig():
    files=['admet_v26_run.py','admet_v26_protocol.json']
    return hashlib.sha256(json.dumps(dict(model=model_signature(),files={n:b.sha(ROOT/n) for n in files}),sort_keys=True).encode()).hexdigest()

def assess(d):
    a=json.loads((b.OUT.parent/'admet_model_outputs_v25/analysis.json').read_text())
    assert a['signature']==model_signature() and not a['selected_tasks']
    cv=np.load(ROOT/'admet_model_outputs_v25/cv_predictions.npy');old=np.load(b.OUT/'cv_predictions.npy')[1]
    valid=np.load(ROOT/'admet_model_outputs_v25/valid_predictions.npy');prior=v25.v24_valid(d)
    f=np.load(b.OUT/'folds.npy');records=[]
    for choice in P['choices']:
        t=choice['task'];k=d['tasks'].index(t);m=d['metrics'][k];variant=choice['variant'];weight=choice['weight']
        gains=[]
        for fold in range(3):
            ix=np.flatnonzero((d['tid']==k)&(f==fold));new=b.blend(m,old[ix],cv[variant,ix],weight)
            gains.append(b.delta(m,d['y'][ix],old[ix],new))
        vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        merged=b.blend(m,prior[vi],valid[variant,vi],weight)
        valid_gain=b.delta(m,d['y'][vi],prior[vi],merged)
        assert np.mean(gains)>0 and sum(g>=0 for g in gains)>=2 and valid_gain>0,(t,gains,valid_gain)
        records.append(dict(**choice,metric=m,fold_gains=gains,mean_cv_gain=float(np.mean(gains)),
                            historical_valid_v24=b.metric(m,d['y'][vi],prior[vi]),
                            historical_valid_mixture=b.metric(m,d['y'][vi],merged),
                            historical_valid_gain=valid_gain,valid_rows=len(vi)))
    info=dict(status='EXPLORATORY_ELIGIBLE_FOR_FINAL_REFIT',signature=runsig(),base_model_signature=model_signature(),
              baseline_sha256=P['baseline_sha256'],records=records,mean_cv_gain=float(np.mean([r['mean_cv_gain'] for r in records])),
              mean_historical_valid_gain=float(np.mean([r['historical_valid_gain'] for r in records])),
              heldout_score_estimate=None,
              limitation='Weights/tasks chosen after reused valid inspection; descriptive CV and valid gains are not independent, not official points; platform score unknown.',
              test_labels_used=False)
    b.save(OUT/'analysis.json',info);b.emit('v26_assess',records=records)

def final(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==runsig()
    fit=np.flatnonzero(d['split']<2);query=np.flatnonzero(d['split']==2)
    selected=[d['tasks'].index(x['task']) for x in P['choices']]
    parts=[train(d,fit,query,seed,ROOT/'admet_model_outputs_v25'/'final_v26'/str(seed),selected=selected) for seed in P['final_seeds']]
    predictions=np.full((2,len(d['y'])),np.nan);predictions[:,query]=np.mean(parts,0)
    np.save(OUT/'test_predictions.npy',predictions)
    compact={t:[predictions[v,(d['tid']==k)&(d['split']==2)].tolist() for v in range(2)] for k,t in enumerate(d['tasks']) if k in selected}
    b.save(OUT/'test_predictions_compact.json',compact)
    export(d)

def export(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==runsig()
    source=next((ROOT/'admet_submissions/v24').glob('*.csv'));assert b.sha(source)==P['baseline_sha256']
    rows=read(source);before=[x.copy() for x in rows];template=read(ROOT/'admet_official/submission_template.csv')
    p=np.load(OUT/'test_predictions.npy');changed={};replay={}
    for record in report['records']:
        t=record['task'];k=d['tasks'].index(t);ix=np.flatnonzero((d['tid']==k)&(d['split']==2))
        pos=[i for i,x in enumerate(rows) if i>0 and x[0]==t];old=np.array([float(rows[i][4]) for i in pos])
        new=b.blend(record['metric'],old,p[record['variant'],ix],record['weight'])
        for i,value in zip(pos,new):rows[i][4]=repr(float(value))
        changed[t]=int(np.sum(old!=new));replay[t]=new
    assert changed and len(rows)==len(template)==16380 and rows[0]==template[0]
    for x,y in zip(rows[1:],template[1:]):
        assert len(x)==6 and x[:4]+x[5:]==y[:4]+y[5:]
        assert np.isfinite(float(x[4]))
        if x[5] in ['AUROC','AUPRC']:assert 0<=float(x[4])<=1
    dest=ROOT/'admet_submissions/v26';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20260928-v26.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows)
    path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'));actual=read(path)
    for t,expected in replay.items():np.testing.assert_array_equal([float(x[4]) for x in actual[1:] if x[0]==t],expected)
    for x,y in zip(actual[1:],before[1:]):
        if x[0] not in changed:assert x==y
    info=dict(status='EXPERIMENTAL_AWAITING_PLATFORM',baseline='v24',baseline_score=50.248759,
              baseline_sha256=b.sha(source),sha256=b.sha(path),rows=16379,columns=6,tasks=22,
              encoding='UTF-8 BOM',line_endings='CRLF',bytes=path.stat().st_size,changed_tasks=changed,
              template_identity_exact=True,model_reload_verified=True,serialized_prediction_replay=True,
              training_host=P['training_host'],local_gpu_training=False,automatically_uploaded=False,
              platform_score=None,signature=runsig(),limitation=report['limitation'])
    b.save(dest/'verification.json',info);b.emit('v26_export',**info)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['assess','final','export'])
    args=parser.parse_args();b.remote_only();assert os.name!='nt' and ROOT==Path(P['remote_root'])
    OUT.mkdir(parents=True,exist_ok=True);b.emit('v26_start',stage_requested=args.stage,signature=runsig())
    d=b.build_data()
    if args.stage=='assess':assess(d)
    elif args.stage=='final':final(d)
    else:export(d)
