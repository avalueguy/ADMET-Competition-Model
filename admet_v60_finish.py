"""Freeze new experts, compare against real v57 proxy, fresh source gates, export once."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
import csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import admet_v24_models as b
import admet_v27_models as graph
import admet_v44_run as pair
import admet_v46_run as adapter
import admet_v58_train as seq
import admet_v59_kernel as kernel
import admet_v60_baseline as anchor

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v60'

def signature():return hashlib.sha256((b.sha(__file__)+b.sha(ROOT/'admet_v60_baseline.py')+seq.signature()+kernel.signature()).encode()).hexdigest()

def inputs():
    start=time.time()
    while time.time()-start<10800:
        data=[]
        for mod in [seq,kernel]:
            p=mod.OUT/'analysis.json'
            if p.exists():
                a=json.loads(p.read_text())
                if a.get('status')=='FROZEN':data.append((mod,a))
        if len(data)==2:return data
        time.sleep(20)
    raise TimeoutError('v58/v59 still incomplete')

def evaluate(d,source_reports,basev,basec):
    folds=np.load(b.OUT/'folds.npy');records=[]
    for mod,report in source_reports:
        assert report['signature']==mod.signature()
        for r in report['tasks']:
            if not r['selection']:continue
            task=r['task'];s=r['selection'];k=d['tasks'].index(task);m=d['metrics'][k]
            tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
            cp=np.load(mod.OUT/task/'cv_predictions.npy')[s['variant']]
            z=np.load(mod.OUT/task/'valid/predictions.npz');np.testing.assert_array_equal(z['query'],vi)
            vp=seq.variants(z['p'],m)[s['variant']] if mod is seq else z['p'][s['variant']]
            gains=[b.delta(m,d['y'][ix],basec[ix],b.blend(m,basec[ix],cp[ix],s['weight'])) for f in range(3) for ix in [tr[folds[tr]==f]]]
            vg=b.delta(m,d['y'][vi],basev[vi],b.blend(m,basev[vi],vp,s['weight']))
            obj=float(.85*np.mean(gains)+.15*vg-.07*np.std(gains))
            record=dict(task=task,metric=m,family='sequence' if mod is seq else 'kernel',selection=s,
                versus_v57=dict(cv_gains=gains,cv_mean=float(np.mean(gains)),valid_gain=vg,objective=obj),
                eligible=bool(np.mean(gains)>.0003 and obj>.0003 and sum(x>0 for x in gains)>=2))
            records.append(record);b.emit('v60_recheck',**record)
    b.save(OUT/'rechecked.json',dict(status='FROZEN',signature=signature(),tasks=records));return records

def fresh(d,views,records):
    folds=np.load(b.OUT/'folds.npy');results=[]
    for record in records:
        if not record['eligible']:continue
        task=record['task'];k=d['tasks'].index(task);m=d['metrics'][k];s=record['selection'];parts=[]
        for repeat,seed in enumerate([601017,601018]):
            if record['family']=='sequence':
                allrows=np.flatnonzero((d['tid']==k)&(d['split']<2));groups=np.unique(d['rg'][allrows])
                held=np.random.default_rng(seed+k).permutation(groups)[:max(1,int(.25*len(groups)))]
                mask=np.isin(d['rg'][allrows],held);rr,q=allrows[~mask],allrows[mask]
                pp=seq.variants(seq.fit(d,views,k,rr,q,seq.OUT/task/f'fresh_{seed}'),m)[s['variant']]
                # matched old source, refit under exactly the same new held groups.
                if m=='MAE':
                    controls=pair.variants(pair.fit(d,k,rr,q,pair.OUT/task/f'v60_control_{seed}'),m);ctrl=controls[0]
                else:
                    # Fixed 0.3/3/30 SV source is not compared across contaminated graph features.
                    from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
                    raw=np.concatenate([d['fp2'],d['fcfp'],np.nan_to_num(d['desc']),d['emb']],1)
                    tf=b.Target(d['y'][rr],m);cls=m in ['AUROC','AUPRC']
                    model=(ExtraTreesClassifier if cls else ExtraTreesRegressor)(n_estimators=400,min_samples_leaf=2,max_features=.6,n_jobs=2,random_state=seed)
                    model.fit(raw[d['mi'][rr]],tf.encode(d['y'][rr]));ctrl=model.predict_proba(raw[d['mi'][q]])[:,1] if cls else model.predict(raw[d['mi'][q]])
                provenance='New scaffold repartition; freshly trained fixed source control, not full historical ensemble'
            else:
                fold=repeat;allq=np.flatnonzero((d['tid']==k)&(d['split']==0)&(folds==fold));groups=np.unique(d['rg'][allq])
                held=np.random.default_rng(seed+k).permutation(groups)[:max(1,int(.8*len(groups)))]
                q=allq[np.isin(d['rg'][allq],held)];rr=np.flatnonzero((d['tid']==k)&(d['split']<2)&~np.isin(d['rg'],held))
                partition='cv/fold'+str(fold);pp=kernel.fit(d,k,rr,q,partition,kernel.OUT/task/f'fresh_{seed}')[s['variant']]
                sourceq,sourcep=adapter.original(d,k,partition);lookup={int(x):j for j,x in enumerate(sourceq)}
                ctrl=sourcep[1,[lookup[int(x)] for x in q]]
                provenance='Resampled 80% of source-excluded fold groups; remaining allowed train+valid rows refitted; original graph-hybrid control, not complete ensemble'
            merged=b.blend(m,ctrl,pp,s['weight']);gain=b.delta(m,d['y'][q],ctrl,merged)
            parts.append(dict(seed=seed,fit_rows=len(rr),query_rows=len(q),control=b.metric(m,d['y'][q],ctrl),
                candidate=b.metric(m,d['y'][q],merged),gain=gain,global_scaffold_overlap=0,test_labels_used=False,provenance=provenance))
            b.emit('v60_fresh',task=task,family=record['family'],**parts[-1])
        gains=[r['gain'] for r in parts];r=dict(**record,fresh=parts,mean_fresh_gain=float(np.mean(gains)),
            pass_gate=bool(np.mean(gains)>0 and min(gains)>-.025))
        results.append(r);b.save(OUT/'fresh.json',dict(status='IN_PROGRESS',tasks=results,signature=signature()))
    b.save(OUT/'fresh.json',dict(status='COMPLETE',tasks=results,signature=signature()));return results

def export(d,views,records):
    candidates={}
    for r in records:
        if r['pass_gate']:candidates.setdefault(r['task'],[]).append(r)
    winners={t:max(rs,key=lambda r:r['versus_v57']['objective']) for t,rs in candidates.items()}
    if not winners:
        b.save(OUT/'no_release.json',dict(status='NO_VERIFIED_GAIN',signature=signature()));return
    source=next((ROOT/'admet_submissions/v57').glob('*.csv'));rows=anchor.rows(source);old=[a.copy() for a in rows];template=anchor.rows(ROOT/'admet_official/submission_template.csv');replay={};changed={}
    for task,r in winners.items():
        k=d['tasks'].index(task);m=d['metrics'][k];s=r['selection'];rr=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2))
        if r['family']=='sequence':
            raw=seq.fit(d,views,k,rr,q,seq.OUT/task/'final');p=seq.variants(raw,m)[s['variant']]
        else:
            raw=kernel.fit(d,k,rr,q,'final',kernel.OUT/task/'final');p=raw[s['variant']]
        pos=[i for i,a in enumerate(rows) if i and a[0]==task];np.testing.assert_array_equal([int(rows[i][1]) for i in pos],d['row'][q])
        base=np.array([float(rows[i][4]) for i in pos]);result=b.blend(m,base,p,s['weight'])
        for i,v in zip(pos,result):rows[i][4]=repr(float(v))
        changed[task]=int(np.sum(result!=base));replay[task]=dict(family=r['family'],metric=m,selection=s,baseline=base.tolist(),branch=p.tolist(),result=result.tolist(),evidence=r)
    assert len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,t in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==t[:4]+t[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v60';dest.mkdir(exist_ok=True);file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261017.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows);raw=b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8')
    if file.exists():assert file.read_bytes()==raw
    else:file.write_bytes(raw)
    b.save(OUT/'fusion_replay.json',replay)
    b.save(dest/'verification.json',dict(status='AWAITING_PLATFORM_EVALUATION',signature=signature(),sha256=b.sha(file),baseline_sha256=b.sha(source),
        rows=16379,columns=6,tasks=22,changed_tasks=changed,bytes=len(raw),encoding='UTF-8 BOM',line_endings='CRLF',
        official_score=None,known_score_upper_bound=50.755035,local_gpu_training=False,automatically_uploaded=False))
    b.save(OUT/'release.json',dict(signature=signature(),selected=winners,sha256=b.sha(file),official_score=None,
        limitations='Development labels repeatedly reused. Full historical ensemble OOF unavailable. Fresh graph source checks resample excluded groups; cannot infer official aggregate score or top-ten position.'))
    b.emit('v60_export',path=str(file),sha256=b.sha(file),changed_tasks=changed)

def main():
    b.remote_only();OUT.mkdir(exist_ok=True);reports=inputs();d=b.build_data();views=seq.tokens(d)
    basev=np.load(OUT/'baseline_valid.npy');basec=np.load(OUT/'baseline_cv_proxy.npy')
    rechecked=evaluate(d,reports,basev,basec);passed=fresh(d,views,rechecked);export(d,views,passed)
    b.save(OUT/'completed.json',dict(status='COMPLETE',signature=signature()))

if __name__=='__main__':main()
