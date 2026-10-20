"""Publish v15, replay trained bases independently, and report all regressions."""
import argparse,csv,io,json,shutil,tarfile
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from admet_v6_runner import remote_only,save,sha,emit,metrics,kind,get_metric
from admet_v15_runner import OUT,ROOT,read,protocol,names,anchor,materialize,candidates
from admet_v15_models import load_task,predict
from admet_v15_ensemble import combine,select,score

OLD=next(Path('admet_submissions/v14').glob('*.csv'))
UPLOAD=Path('admet_submissions/v15')/OLD.name


def publish(task):
    dest=ROOT/task;dest.mkdir(parents=True,exist_ok=True);locked=dest/'selection.json'
    if locked.exists():assert read(locked)['protocol_sha256']==sha('admet_v15_protocol.json');return
    decision=read(OUT/task/'decision.json');s=decision['selection'];prior=Path('admet_release_v14')/task
    if decision['changed']:
        d=load_task(task);n=len(d['frames']['train'])+len(d['frames']['valid']);query=np.arange(n,len(d['smiles']))
        old=pd.read_csv(prior/f'submission_{task}.csv',dtype={'Drug_ID':str},keep_default_na=False);b=old.Y.to_numpy(float)
        active=[i for i,w in enumerate(s['weights'][1:]) if w>0];cc=[b.copy() for _ in s['names']]
        values=materialize(task,d,np.arange(n),query,[s['names'][i] for i in active],'final')
        for i,c in zip(active,values):cc[i]=c
        q=combine(b,cc,s['weights'],s['mode'],s['metric']);assert np.isfinite(q).all()
        if s['metric'] in ['AUROC','AUPRC']:assert ((q>=0)&(q<=1)).all()
        pd.DataFrame(dict(Drug_ID=old.Drug_ID,Y=q)).to_csv(dest/f'submission_{task}.csv',index=False)
        valid=pd.read_csv(OUT/task/'validation_comparison.csv')
        pd.DataFrame(dict(Y_true=valid.Y_true,Y_pred=valid.blend)).to_csv(dest/'validation_predictions.csv',index=False)
    else:
        for f in [f'submission_{task}.csv','validation_predictions.csv']:shutil.copy2(prior/f,dest/f)
    decision.update(prediction_sha256=sha(dest/f'submission_{task}.csv'),prior_prediction_sha256=sha(prior/f'submission_{task}.csv'))
    save(locked,decision);emit(stage='v15_publish',task=task,changed=decision['changed'])


def export():
    assert sha(OLD)==protocol()['baseline_sha256'];routes=[read(ROOT/t/'selection.json') for t in names()]
    rows=list(csv.reader(io.StringIO(OLD.read_bytes().decode('utf-8-sig'),newline='')));original=[r.copy() for r in rows]
    changed={r['task'] for r in routes if r['changed']}
    preds={t:pd.read_csv(ROOT/t/f'submission_{t}.csv').Y.to_numpy(float) for t in changed}
    for row in rows[1:]:
        if row[0] in changed:row[4]=repr(float(preds[row[0]][int(row[1])]))
    for a,b in zip(original,rows):
        assert a[:4]+a[5:]==b[:4]+b[5:]
        if a[0] not in changed:assert a==b
    out=io.StringIO(newline='');csv.writer(out,lineterminator='\r\n').writerows(rows)
    blob=b'\xef\xbb\xbf'+out.getvalue().encode();UPLOAD.parent.mkdir(parents=True,exist_ok=True)
    if UPLOAD.exists():assert UPLOAD.read_bytes()==blob
    UPLOAD.write_bytes(blob);save(ROOT/'routing.json',routes)
    save(ROOT/'verification.json',dict(status='EXPORTED_AWAITING_REPLAY',submission=str(UPLOAD),sha256=sha(UPLOAD),
        changed=sorted(changed),unchanged=sorted(set(names())-changed),baseline_score=49.419624,platform_score=None,
        next_target=50.019624,ultimate_target=51.584929,rows=16379,tasks=22,columns=6,
        encoding='UTF-8 BOM',line_endings='CRLF',automatic_submission=False,per_metric_nonregression_required=False))


def replay(task,stage,d,expected_fit,expected_query):
    root=OUT/task/stage;out={};labels=pd.concat([d['frames']['train'],d['frames']['valid']],ignore_index=True).Y.to_numpy(float)
    import hashlib
    for meta in sorted(root.glob('*.json')):
        m=read(meta);path=meta.with_suffix('.joblib');assert sha(path)==m['sha256']
        bundle=joblib.load(path);np.testing.assert_array_equal(bundle['fit_rows'],expected_fit)
        assert not set(expected_fit)&set(expected_query)
        assert bundle['fit_target_sha256']==hashlib.sha256(np.asarray(labels[expected_fit],dtype='<f8').tobytes()).hexdigest()
        with np.load(meta.with_suffix('.npz')) as z:
            np.testing.assert_array_equal(z['fit'],expected_fit);np.testing.assert_array_equal(z['query'],expected_query);cached=z['prediction']
        q=predict(bundle,d,expected_query);np.testing.assert_allclose(q,cached,atol=5e-6,rtol=5e-6)
        for audit in bundle['model'].get('source_audit',[]):assert audit['heldout_molecule_overlap']==audit['heldout_scaffold_overlap']==0
        out[bundle['name']]=q;del bundle
    return out


def verify():
    v=read(ROOT/'verification.json');assert sha(UPLOAD)==v['sha256'] and sha(OLD)==protocol()['baseline_sha256']
    for f,h in read('admet_v15_source_freeze.json').items():assert sha(f)==h,f
    blob=UPLOAD.read_bytes();assert blob.startswith(b'\xef\xbb\xbf') and blob.count(b'\r\n')==16380
    assert blob.count(b'task,sample_idx,drug_id,smiles,y_pred,metric')==1
    frame=pd.read_csv(UPLOAD,dtype=str,keep_default_na=False);old=pd.read_csv(OLD,dtype=str,keep_default_na=False)
    template=pd.read_csv('admet_official/submission_template.csv',dtype=str,keep_default_na=False)
    assert len(frame)==16379 and frame.task.nunique()==22 and frame.drop(columns='y_pred').equals(template.drop(columns='y_pred'))
    task_checks=[]
    for t in names():
        r=read(ROOT/t/'selection.json');s=r['selection'];d=load_task(t);n=len(d['frames']['train']);nv=len(d['frames']['valid'])
        cv=replay(t,'valid',d,np.arange(n),np.arange(n,n+nv))
        saved=pd.read_csv(OUT/t/'validation_comparison.csv');baseline=pd.read_csv(Path('admet_release_v14')/t/'validation_predictions.csv')
        np.testing.assert_allclose(saved.Y_true,d['frames']['valid'].Y);np.testing.assert_allclose(saved.baseline,baseline.Y_pred)
        for i,name in enumerate(s['names']):np.testing.assert_allclose(saved[f'candidate_{i}'],cv[name],atol=5e-6,rtol=5e-6)
        reconstructed=combine(saved.baseline,[cv[name] for name in s['names']],s['weights'],s['mode'],s['metric'])
        np.testing.assert_allclose(saved.blend,reconstructed,atol=5e-6,rtol=5e-6)
        b=anchor(t);pool,_=candidates(t,b);best,_=select(b.Y_true.to_numpy(),b.fold.to_numpy(),b.Y_pred.to_numpy(),[pool[k] for k in s['names']],
            saved.Y_true.to_numpy(),saved.baseline.to_numpy(),[saved[f'candidate_{i}'].to_numpy() for i in range(len(s['names']))],s['metric'],protocol())
        assert best['weights']==s['weights'] and best['mode']==s['mode'];assert best['objective']==s['objective'] or abs(best['objective']-s['objective'])<1e-10
        q=pd.read_csv(ROOT/t/f'submission_{t}.csv').Y.to_numpy();assert sha(ROOT/t/f'submission_{t}.csv')==r['prediction_sha256']
        if r['changed']:
            total=n+nv;final=replay(t,'final',d,np.arange(total),np.arange(total,len(d['smiles'])))
            btest=pd.read_csv(Path('admet_release_v14')/t/f'submission_{t}.csv').Y.to_numpy()
            expected=combine(btest,[final.get(name,btest) for name in s['names']],s['weights'],s['mode'],s['metric'])
            np.testing.assert_allclose(expected,q,atol=5e-6,rtol=5e-6)
            it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=692509).split(np.arange(n),groups=d['groups'][:n]))
            rep=replay(t,'repeat',d,it,iv);savedr=pd.read_csv(OUT/t/'repeat_comparison.csv');np.testing.assert_array_equal(savedr.row,iv)
            expected=combine(savedr.baseline,[rep.get(name,savedr.baseline) for name in s['names']],s['weights'],s['mode'],s['metric'])
            np.testing.assert_allclose(savedr.blend,expected,atol=5e-6,rtol=5e-6)
        else:
            assert frame.loc[frame.task.eq(t),'y_pred'].tolist()==old.loc[old.task.eq(t),'y_pred'].tolist()
        np.testing.assert_allclose(frame.loc[frame.task.eq(t),'y_pred'].astype(float),q,atol=1e-12,rtol=1e-12)
        assert np.isfinite(q).all()
        if s['metric'] in ['AUROC','AUPRC']:assert ((q>=0)&(q<=1)).all()
        task_checks.append(dict(task=t,changed=r['changed'],passed=True));emit(stage='v15_replay',task=t,passed=True)
    v.update(status='VERIFIED',independent_replay=True,task_checks=task_checks)
    save(ROOT/'independent_verification.json',v)


def report():
    routes=read(ROOT/'routing.json');rows=[];diagnostics=[]
    for r in routes:
        s=r['selection'];row=dict(task=r['task'],changed=r['changed'],metric=s['metric'],mode=s['mode'],names=s['names'],weights=s['weights'],
            oof_primary_gain=s['mean_oof_gain'],valid_primary_gain=s['valid_gain'],objective=s['objective'],repeat_primary_gain=r['repeat'].get('primary_gain'),
            **{'before_'+k:v for k,v in r['before'].items()},**{'after_'+k:v for k,v in r['after'].items()})
        rows.append(row)
        for check in r['diagnostics']:
            q={k:v for k,v in check.items() if k not in ['before','after']}
            for label in ['before','after']:q.update({label+'_'+k:v for k,v in check.get(label,{}).items()})
            diagnostics.append(q)
    pd.DataFrame(rows).to_csv('admet_diagnostics/v15_comparison.csv',index=False)
    pd.DataFrame(diagnostics).to_csv('admet_diagnostics/v15_all_diagnostics.csv',index=False)
    aggregate=dict(task_count=22,changed=sum(r['changed'] for r in rows),mean_oof_primary_surrogate=float(np.mean([r['oof_primary_gain'] for r in rows])),
        mean_valid_primary_surrogate=float(np.mean([r['valid_primary_gain'] for r in rows])),
        valid_primary_regression_tasks=[r['task'] for r in rows if r['valid_primary_gain']<0],
        distant_regressions=[r for r in diagnostics if r['group']=='distant' and r.get('primary_gain',0)<0],
        formula='Mean relative MAE improvement or absolute ranking-metric improvement, equal tasks. Not official score.',
        platform_score=None,baseline_score=49.419624,requested_target=50.019624,
        caveat='Validation used for selection; historical folds reused; no unbiased hidden-score estimate or guarantee.')
    save('admet_diagnostics/v15_aggregate.json',aggregate)
    with tarfile.open('admet_v15_review.tar.gz','w:gz') as archive:
        files=[p for root in [ROOT,OUT] for p in root.rglob('*') if p.is_file() and p.suffix in ['.json','.csv']]
        files+=[UPLOAD]+list(Path('admet_diagnostics').glob('v15_*'))+list(Path('logs').glob('v15*.log'))
        for p in files:archive.add(p,arcname=str(p),recursive=False)
    emit(stage='v15_packaged',changed=aggregate['changed'],bytes=Path('admet_v15_review.tar.gz').stat().st_size)


if __name__=='__main__':
    remote_only();ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['publish','export','verify','report']);ap.add_argument('--tasks',nargs='*');args=ap.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='publish':
            for t in args.tasks or names():publish(t)
        else:globals()[args.stage]()
