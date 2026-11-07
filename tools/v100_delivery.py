"""Template-exact conformer release and independent model replay."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,csv,io,json,sys,tarfile
from collections import Counter
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v100_shape_kernel as v
from verify_submission import verify
from repack_upload_template import field_spans


def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))


def selected(require=True):
    f=json.loads((v.OUT/'frozen_selection.json').read_text());c=json.loads((v.OUT/'confirmation.json').read_text())
    assert f['signature']==c['signature']==v.sig() and c['status']=='COMPLETE'
    assert c['frozen_sha256']==v.b.sha(v.OUT/'frozen_selection.json')
    rs=[r for r in c['records'] if r['accepted']];assert [r['task'] for r in rs]==c['accepted']
    if require:assert rs,'No eligible expert: no CSV.'
    for r in rs:
        assert all(x['gain']>0 for x in r['confirmation']) and np.mean([x['gain'] for x in r['confirmation']])>v.threshold(r['metric'])
        assert any(all(r[key]==ch[key] for key in ('task','metric','config','weight')) for ch in f['selected'])
    return rs


def export():
    rs=selected();d=v.load();old=read(v.ANCHOR);rows=[r.copy() for r in old];changed={}
    for r in rs:
        k=d['tasks'].index(r['task']);p=v.mix(r['metric'],v.base(d,k,'final'),v.expert(d,k,'final',r['config']),r['weight'])
        pos=[i for i,row in enumerate(old) if i and row[0]==r['task']];assert len(p)==len(pos) and np.isfinite(p).all()
        for i,x in zip(pos,p):rows[i][4]=repr(float(x))
        changed[r['task']]=sum(rows[i][4]!=old[i][4] for i in pos)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();ss=field_spans(raw);chunks=[];cursor=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=ss[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks)
    file=ROOT/'admet_submissions/v100_shape_kernel/我从山北来带着兰花草-小分子药物预测技术创新赛-20261026.csv';file.parent.mkdir(parents=True,exist_ok=True)
    if file.exists():assert file.read_bytes()==output,'Refuse overwrite of different release.'
    else:file.write_bytes(output)
    fmt=verify(file,template)
    v.b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v100',file=file.relative_to(ROOT).as_posix(),sha256=v.b.sha(file),
        anchor='v94',anchor_sha256=v.ANCHOR_SHA,anchor_official_score_user_reported=50.916,selected=rs,changed_tasks=changed,unchanged_tasks=22-len(changed),
        verification=fmt,official_score=None,test_labels_used=False,external_data_new_branch=False,new_pretrained_weights=False,
        limitations=['Historical proxies not exact v94 OOF; labels reused historically.','Synthetic conformers approximate molecular shape; joint view uses conditional official assays.',
                    'Historical baseline includes early pretrained sources.','No estimate or guarantee of official score gain.']))
    v.emit(stage='export',changed=changed,sha256=v.b.sha(file))


def cpu_shape(d,rows,fit,a):
    fi=d['mi'][fit];qi=d['mi'][rows];out=np.zeros((len(rows),len(fit)))
    for j,raw in enumerate((d['views']['physchem'],d['conformer_features'])):
        train=raw[fi].astype(float);query=raw[qi].astype(float)
        center=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in train.T])
        full=np.where(np.isfinite(train),train,center);scale=full.std(0);active=np.flatnonzero(scale>1e-5);scale=scale.clip(.05)
        for key,val in (('center',center),('scale',scale),('active',active)):np.testing.assert_array_equal(a[f'view{j}_{key}'],val)
        if not len(active):continue
        tr=((full[:,active]-center[active])/scale[active]).clip(-6,6).astype(np.float32).astype(float)
        qe=((np.where(np.isfinite(query),query,center)[:,active]-center[active])/scale[active]).clip(-6,6).astype(np.float32).astype(float)
        norm=np.square(tr).sum(1)
        for lo in range(0,len(rows),128):
            x=qe[lo:lo+128];dist=np.maximum(np.square(x).sum(1)[:,None]+norm[None]-2*x@tr.T,0)/len(active)
            out[lo:lo+len(x)]+=(.5*np.exp(-.5*dist)+.5*np.exp(-2*dist))/2
    return out

def audit(d,k,part):
    import v92_delivery as old
    file=v.OUT/'models'/d['tasks'][k]/part/'predictions.npz';a=dict(np.load(file));meta=json.loads(file.with_name('metadata.json').read_text())
    assert str(a['signature'])==meta['signature']==v.sig() and v.b.sha(file)==meta['sha256']
    expected=old.state(d,k,part)
    for key,x in expected.items():np.testing.assert_array_equal(a[key],x)
    np.testing.assert_array_equal(a['alphas'],v.ALPHAS);fit,q=a['fit'],a['query']
    assert not set(fit)&set(q)
    if part!='final':
        assert not set(d['groups'][fit])&set(d['groups'][q])
        saved=d['y'][q].copy();d['y'][q]=1e6;perturbed=v.state(d,k,part);d['y'][q]=saved
        for key,x in expected.items():np.testing.assert_array_equal(perturbed[key],x)
    chemical=old.cpu_kernel(d,q,fit,'multi');shape=cpu_shape(d,q,fit,a)
    query={'sum':.5*chemical+.5*shape,'product':chemical*shape}
    ids=np.unique(np.linspace(0,len(fit)-1,min(17,len(fit)),dtype=int))
    chem=old.cpu_kernel(d,fit[ids],fit,'multi');sh=cpu_shape(d,fit[ids],fit,a);train={'sum':.5*chem+.5*sh,'product':chem*sh}
    errors=[];differences=[]
    for i,view in enumerate(('sum','product')):
        for j,alpha in enumerate(v.ALPHAS):
            coef=a['coefficients'][i,j].astype(float);p=query[view]@coef
            np.testing.assert_allclose(a['values'][i,j],p,atol=5e-4,rtol=5e-4)
            error=float(np.max(np.abs(train[view]@coef+alpha*coef[ids]-a['target'][ids])))
            assert error<.006,(part,view,alpha,error)
            errors.append(error);differences.append(float(np.max(np.abs(p-a['values'][i,j]))))
    rec=dict(task=d['tasks'][k],part=part,status='PASS',sha256=v.b.sha(file),query_rows=len(q),cpu_prediction_max_difference=max(differences),
        cpu_sampled_system_max_residual=max(errors),descriptor_moments_fit_only=True,target_query_label_perturbation=part!='final')
    v.emit(stage='cpu_replay',**rec);return a,rec

def replay():
    rs=selected(False);d=v.load();records=[];choices=rs or json.loads((v.OUT/'frozen_selection.json').read_text())['selected']
    if choices:keys=[(r['task'],p) for r in choices for p in (('fold0','fold1','fold2','valid','final') if rs else ('fold0','fold1','fold2','valid'))]
    else:keys=[(task,'fold0') for task in d['tasks']]
    for task,part in keys:
        _,rec=audit(d,d['tasks'].index(task),part);records.append(rec)
    report=dict(status='PASS',records=records,no_csv=not bool(rs),test_labels_used=False)
    if rs:
        rel=json.loads((v.OUT/'release.json').read_text());file=ROOT/rel['file'];assert v.b.sha(file)==rel['sha256'];old,actual=read(v.ANCHOR),read(file)
        for r in rs:
            k=d['tasks'].index(r['task'])
            for part in ('fold0','fold1','fold2','valid','final'):
                p=v.mix(r['metric'],v.base(d,k,part),v.expert(d,k,part,r['config']),r['weight'])
                if part=='final':np.testing.assert_allclose(p,[float(z[4]) for z in actual[1:] if z[0]==r['task']],atol=1e-12,rtol=1e-12)
                else:
                    q=v.r83.split(d,k,part)[1];record=next(x for x in r['confirmation'] if x['part']==part)
                    np.testing.assert_allclose(v.r83.score(r['metric'],d['y'][q],p),record['candidate'],atol=1e-12,rtol=1e-12)
        changed=Counter()
        for a,z in zip(old[1:],actual[1:]):
            assert a[:4]+a[5:]==z[:4]+z[5:]
            if a[4]!=z[4]:changed[z[0]]+=1
            if z[0] not in rel['changed_tasks']:assert a==z
        assert dict(changed)==rel['changed_tasks'];template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template)
        raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);chunks=[];cursor=0
        for i in range(1,len(actual)):
            lo,hi=ss[i][4];a,z=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
        chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
        fmt.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),models=len(records),cpu_prediction_replay='PASS',source_reconstruction='PASS')
        v.b.save(file.with_suffix('.verification.json'),fmt);report['submission']=fmt;rel.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt)
    v.b.save(v.OUT/'model_verification.json',report)
    if rs:rel['verification_sha256']=v.b.sha(v.OUT/'model_verification.json');v.b.save(v.OUT/'release.json',rel)
    v.emit(stage='verification',status='PASS',models=len(records),no_csv=not bool(rs))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['export','verify']);args=ap.parse_args()
    dict(export=export,verify=replay)[args.stage]()

