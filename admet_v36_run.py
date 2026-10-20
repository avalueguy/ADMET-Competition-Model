"""Freeze and replay one reviewable release from complementary remote model sources."""
import csv,io,json,hashlib
from pathlib import Path
import numpy as np
import admet_v24_models as b
import admet_v33_run as n
import admet_v34_run as kmodel
import admet_v35_run as xmodel
ROOT=n.ROOT;OUT=ROOT/'admet_model_outputs_v36';P=json.loads((ROOT/'admet_v36_protocol.json').read_text())
SOURCES={33:n,34:kmodel,35:xmodel}
def signature():
    return hashlib.sha256(json.dumps(dict(sources={str(v):module.signature() for v,module in SOURCES.items()},
        files={f:b.sha(ROOT/f) for f in ['admet_v36_run.py','admet_v36_protocol.json']}),sort_keys=True).encode()).hexdigest()
def source_predictions(d,t,source,variant,partition):
    module=SOURCES[source];z=np.load(module.OUT/t/partition/'predictions.npz')
    assert str(z['signature'])==module.signature()
    m=d['metrics'][d['tasks'].index(t)];return module.variants(z['p'],m)[variant]
def evaluate(d):
    valid=np.load(n.OUT/'baseline_valid.npy');cv=np.load(n.OUT/'baseline_cv_proxy.npy');folds=np.load(b.OUT/'folds.npy')
    reports={v:json.loads((module.OUT/'analysis.json').read_text()) for v,module in SOURCES.items()}
    for v,report in reports.items():assert report['status']=='EVALUATED' and report['signature']==SOURCES[v].signature()
    records=[]
    for k,t in enumerate(d['tasks']):
        m=d['metrics'][k];vi=np.flatnonzero((d['tid']==k)&(d['split']==1));tr=np.flatnonzero((d['tid']==k)&(d['split']==0));candidates=[]
        for version,report in reports.items():
            prior=next((r for r in report['tasks'] if r['task']==t),None)
            if prior is None or prior['selection'] is None:continue
            variant=prior['selection']['variant'];vp=source_predictions(d,t,version,variant,'valid')
            cp=np.load(SOURCES[version].OUT/t/'cv_predictions.npy')[variant,tr]
            candidates.append(dict(name=f'v{version}',sources=[dict(version=version,variant=variant,weight=1.)],valid=vp,cv=cp))
        if len(candidates)>1:
            vv=[];cc=[]
            for candidate in candidates:
                v=candidate['valid'].copy();c=candidate['cv'].copy()
                if m!='MAE':
                    v=b.rank(v)
                    for fold in range(3):
                        ix=folds[tr]==fold;c[ix]=b.rank(c[ix])
                vv.append(v);cc.append(c)
            candidates.append(dict(name='complement_equal',sources=[dict(**c['sources'][0],) for c in candidates],
                valid=np.mean(vv,0),cv=np.mean(cc,0)))
            for source in candidates[-1]['sources']:source['weight']=1/len(vv)
        grid=[]
        for candidate in candidates:
            for weight in P['weights']:
                gains=[]
                for fold in range(3):
                    active=folds[tr]==fold;ix=tr[active]
                    gains.append(b.delta(m,d['y'][ix],cv[ix],b.blend(m,cv[ix],candidate['cv'][active],weight)))
                vg=b.delta(m,d['y'][vi],valid[vi],b.blend(m,valid[vi],candidate['valid'],weight))
                obj=float(.55*vg+.45*np.mean(gains)-.04*np.std(gains))
                grid.append(dict(name=candidate['name'],sources=candidate['sources'],weight=weight,valid_gain=vg,
                    cv_gains=gains,cv_mean=float(np.mean(gains)),objective=obj,eligible=bool(vg>2e-4 and np.mean(gains)>0 and obj>5e-4)))
        choices=[r for r in grid if r['eligible']];sel=max(choices,key=lambda g:g['objective']) if choices else None
        records.append(dict(task=t,metric=m,valid_baseline=b.metric(m,d['y'][vi],valid[vi]),selection=sel,grid=grid))
    b.save(OUT/'analysis.json',dict(signature=signature(),status='EVALUATED',baseline='v32',baseline_score=50.399027,
        tasks=records,selected_tasks=[r['task'] for r in records if r['selection']],official_score_prediction=None,limitations=P['limitations']))
def export(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    baseline_file=next((ROOT/'admet_submissions/v32').glob('*.csv'));assert b.sha(baseline_file)==P['baseline_sha256']
    def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
    rows=read(baseline_file);template=read(ROOT/'admet_official/submission_template.csv');compact={};changed={}
    for r in report['tasks']:
        s=r['selection']
        if s is None:continue
        t=r['task'];pos=[i for i,row in enumerate(rows) if i>0 and row[0]==t];values=[];rawparts=[]
        for source in s['sources']:
            pred=source_predictions(d,t,source['version'],source['variant'],'final')
            rawparts.append(dict(**source,predictions=pred.tolist()))
            if s['name']=='complement_equal' and r['metric']!='MAE':pred=b.rank(pred)
            values.append(source['weight']*pred)
        compact[t]=rawparts;p=np.sum(values,0);old=np.array([float(rows[i][4]) for i in pos]);assert len(p)==len(old)
        merged=b.blend(r['metric'],old,p,s['weight'])
        for i,value in zip(pos,merged):rows[i][4]=repr(float(value))
        changed[t]=int(np.sum(merged!=old))
    assert changed and len(rows)==len(template)==16380 and rows[0]==template[0]
    for row,t in zip(rows[1:],template[1:]):
        assert len(row)==6 and row[:4]+row[5:]==t[:4]+t[5:] and np.isfinite(float(row[4]))
        if row[5] in ['AUROC','AUPRC']:assert 0<=float(row[4])<=1
    dest=ROOT/'admet_submissions/v36';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261006.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows);path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'));assert read(path)==rows
    b.save(OUT/'test_predictions_compact.json',compact)
    names=['admet_v36_run.py','admet_v36_protocol.json','admet_v33_run.py','admet_v33_protocol.json','admet_v34_run.py','admet_v34_protocol.json','admet_v35_run.py','admet_v35_protocol.json','admet_v36_remote_verify.py']
    b.save(OUT/'source_manifest.json',{name:b.sha(ROOT/name) for name in names})
    b.save(dest/'verification.json',dict(status='EXPERIMENTAL_AWAITING_PLATFORM',signature=signature(),baseline='v32',baseline_score=50.399027,
        baseline_sha256=b.sha(baseline_file),sha256=b.sha(path),rows=16379,tasks=22,columns=6,changed_tasks=changed,bytes=path.stat().st_size,
        template_identity_exact=True,encoding='UTF-8 BOM',line_endings='CRLF',training_host=P['training_host'],local_gpu_training=False,
        platform_score=None,automatically_uploaded=False,limitations=P['limitations']))
    b.emit('v36_export',sha256=b.sha(path),changed_tasks=changed)
if __name__=='__main__':
    b.remote_only();OUT.mkdir(parents=True,exist_ok=True);d=b.build_data();evaluate(d);export(d)
