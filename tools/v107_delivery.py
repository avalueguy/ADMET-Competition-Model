"""Independent replay and exact-template assembly for v102-v104 candidates."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,csv,io,json,sys
from collections import Counter
from pathlib import Path
import numpy as np,joblib,xgboost as xgb
from scipy import sparse
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v102_count_margin as count
import v103_pair_differences as pair
import v104_sparse_margin as unfolded
import v106_sparse_boost as boosted
from verify_submission import verify
from repack_upload_template import field_spans
BRANCHES={102:count,103:pair,104:unfolded,106:boosted};OUT=ROOT/'admet_model_outputs_v107'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def table(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def selected(v):
    f=read(v.OUT/'frozen_selection.json');c=read(v.OUT/'confirmation.json')
    assert f['signature']==c['signature']==v.sig() and c['status']=='COMPLETE'
    assert c['frozen_sha256']==v.b.sha(v.OUT/'frozen_selection.json')
    rs=[r for r in c['records'] if r['accepted']];assert [r['task'] for r in rs]==c['accepted']
    for r in rs:
        assert all(x['gain']>0 for x in r['confirmation']) and np.mean([x['gain'] for x in r['confirmation']])>v.threshold(r['metric'])
        assert any(all(ch[key]==r[key] for key in ('task','metric','config','weight','objective')) for ch in f['selected'])
    return rs

def cpu_count(d,rows,fit,view):
    # Separate CPU double-precision reconstruction from the original log cache.
    output=np.zeros((len(rows),len(fit)));slices=range(3) if view=='circular' else range(6)
    for j in slices:
        raw=d['views']['multiview'][:,j*2048:(j+1)*2048]
        ref=np.rint(np.expm1(raw[d['mi'][fit]])).astype(float);query=np.rint(np.expm1(raw[d['mi'][rows]])).astype(float)
        norm=np.square(ref).sum(1)
        for lo in range(0,len(rows),128):
            x=query[lo:lo+128];dot=x@ref.T;t=np.clip(dot/np.maximum(np.square(x).sum(1)[:,None]+norm[None]-dot,1e-8),0,1)
            output[lo:lo+len(x)]+=(.5*t+.5*t*t)/len(slices)
    if view=='hybrid':
        raw=d['views']['physchem'];tr=raw[d['mi'][fit]].astype(float);q=raw[d['mi'][rows]].astype(float)
        median=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in tr.T])
        full=np.where(np.isfinite(tr),tr,median);scale=full.std(0);active=np.flatnonzero(scale>1e-5);scale=scale.clip(.05)
        tx=((full[:,active]-median[active])/scale[active]).clip(-6,6).astype(np.float32).astype(float)
        qx=((np.where(np.isfinite(q),q,median)[:,active]-median[active])/scale[active]).clip(-6,6).astype(np.float32).astype(float)
        norm=np.square(tx).sum(1);ph=np.zeros_like(output)
        for lo in range(0,len(rows),128):
            x=qx[lo:lo+128];dist=np.maximum(np.square(x).sum(1)[:,None]+norm[None]-2*x@tx.T,0)/max(1,len(active))
            ph[lo:lo+len(x)]=.5*np.exp(-.5*dist)+.5*np.exp(-2*dist)
        output=.8*output+.2*ph
    return output

def cpu_unfolded(d,rows,fit,view):
    out=np.zeros((len(rows),len(fit)))
    for raw in d['unfolded_fingerprints']:
        ref=raw[d['mi'][fit]].astype(np.float64);query=raw[d['mi'][rows]].astype(np.float64)
        if view=='binary_rbf4':ref.data[:]=1;query.data[:]=1
        rn=np.asarray(ref.power(2).sum(axis=1)).ravel();qn=np.asarray(query.power(2).sum(axis=1)).ravel()
        for lo in range(0,len(rows),137):
            dot=(query[lo:lo+137]@ref.T).toarray();t=np.clip(dot/np.maximum(qn[lo:lo+len(dot),None]+rn[None]-dot,1e-10),0,1)
            out[lo:lo+len(dot)]+=t/2
    return .5*out+.5*out**2 if view=='count_t2' else np.exp(4*(out-1))

def audit_margin(v,d,r,part):
    k=d['tasks'].index(r['task']);cfg=r['config'];v.expert(d,k,part,cfg)
    folder=v.OUT/'models'/r['task']/part/f"{cfg['view']}_{cfg['mode']}_c{cfg['C']}"
    meta=read(folder/'metadata.json');a=dict(np.load(folder/'predictions.npz'));pkey='predictions_sha256' if v is count else 'prediction_sha256'
    assert meta['signature']==str(a['signature'])==v.sig() and v.b.sha(folder/'predictions.npz')==meta[pkey]
    assert v.b.sha(folder/'model.joblib')==meta['model_sha256'];model=joblib.load(folder/'model.joblib')
    import v92_delivery as independent
    expected=independent.state(d,k,part)
    for key,x in expected.items():np.testing.assert_array_equal(a[key],x)
    fit,q=a['fit'],a['query'];assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    if part!='final':
        assert not set(d['groups'][fit])&set(d['groups'][q]);saved=d['y'][q].copy();d['y'][q]=1e6
        perturbed=count.state(d,k,part);d['y'][q]=saved
        for key,x in expected.items():np.testing.assert_array_equal(x,perturbed[key])
    fn=cpu_count if v is count else cpu_unfolded
    matrix=fn(d,q,fit,cfg['view']);cls=r['metric'] in ('AUROC','AUPRC')
    replay=(model.decision_function if cls else model.predict)(matrix)
    manual=matrix[:,model.support_]@model.dual_coef_[0]+model.intercept_[0]
    np.testing.assert_allclose(replay,manual,atol=2e-10,rtol=2e-10)
    tol=4e-4 if v is count else 2e-9;np.testing.assert_allclose(a['raw'],replay,atol=tol,rtol=tol)
    ids=np.unique([0,len(q)//2,len(q)-1]);sub=(model.decision_function if cls else model.predict)(matrix[ids])
    np.testing.assert_allclose(replay[ids],sub,atol=1e-10,rtol=1e-10)
    result=dict(task=r['task'],part=part,status='PASS',model_sha256=meta['model_sha256'],cpu_raw_max_difference=float(np.max(np.abs(a['raw']-replay))),
        manual_support_replay=True,subset_prediction=True,query_label_perturbation=part!='final')
    v.emit(stage='independent_replay',**result);return result

def pair_neighbors_cpu(d,a,rows):
    fit=a['fit'];raw=d['views']['multiview'];output=np.zeros((len(rows),len(fit)))
    for j in range(3):
        ref=sparse.csr_matrix(np.rint(np.expm1(raw[d['mi'][fit],j*2048:(j+1)*2048]))).astype(float)
        query=sparse.csr_matrix(np.rint(np.expm1(raw[d['mi'][rows],j*2048:(j+1)*2048]))).astype(float)
        dot=(query@ref.T).toarray();rn=np.asarray(ref.multiply(ref).sum(1)).ravel();qn=np.asarray(query.multiply(query).sum(1)).ravel()
        output+=np.clip(dot/np.maximum(qn[:,None]+rn[None]-dot,1e-8),0,1)/3
    exact=d['canonical'][rows,None]==d['canonical'][fit][None];output[exact]=-1;return output

def audit_pair(d,r,part):
    k=d['tasks'].index(r['task']);cfg=r['config'];pair.expert(d,k,part,cfg)
    folder=pair.OUT/'models'/r['task']/part/f"depth{cfg['depth']}";meta=read(folder/'metadata.json');z=dict(np.load(folder/'predictions.npz'))
    assert str(z['signature'])==meta['signature']==pair.sig() and bsha(folder/'predictions.npz')==meta['prediction_sha256']
    assert bsha(folder/'model.ubj')==meta['model_sha256'] and bsha(folder/'input_state.npz')==meta['input_state_sha256']
    a=pair.prepare(d,k,part);cached=dict(np.load(folder/'input_state.npz'))
    for key,x in a.items():
        if key!='features':np.testing.assert_array_equal(x,cached[key])
    fit,q=a['fit'],a['query'];assert not set(fit)&set(q)
    if part!='final':
        assert not set(d['groups'][fit])&set(d['groups'][q]);saved=d['y'][q].copy();d['y'][q]=1e6;changed=pair.prepare(d,k,part);d['y'][q]=saved
        for key in a:np.testing.assert_array_equal(a[key],changed[key])
    # Independent descriptor standardization and deterministic random projection.
    x=d['views']['physchem'];tr=x[d['mi'][fit]].astype(float)
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in tr.T]);full=np.where(np.isfinite(tr),tr,med)
    std=full.std(0);active=np.flatnonzero(std>1e-5);std=std.clip(.05)
    np.testing.assert_array_equal(med,a['descriptor_center']);np.testing.assert_array_equal(std,a['descriptor_scale']);np.testing.assert_array_equal(active,a['active'])
    allx=x[d['mi'][a['rows']]].astype(float);desc=((np.where(np.isfinite(allx),allx,med)[:,active]-med[active])/std[active]).clip(-6,6).astype(np.float32)
    rp=sparse.csr_matrix(np.rint(np.expm1(d['views']['multiview'][:,:6144])));rp.data=np.sqrt(rp.data)
    proj=(rp@np.random.default_rng(pair.SEED).normal(0,1/8,(6144,64)).astype(np.float32)).astype(np.float32)
    center=proj[d['mi'][fit]].mean(0);scale=proj[d['mi'][fit]].std(0).clip(.05)
    np.testing.assert_array_equal(a['projection_center'],center);np.testing.assert_array_equal(a['projection_scale'],scale)
    expected=np.c_[desc,((proj[d['mi'][a['rows']]]-center)/scale).clip(-6,6).astype(np.float32)]
    np.testing.assert_array_equal(a['features'],expected)
    sample=np.unique(np.r_[np.linspace(0,len(fit)-1,min(8,len(fit)),dtype=int),len(fit)+np.linspace(0,len(q)-1,min(8,len(q)),dtype=int)])
    sim=pair_neighbors_cpu(d,a,a['rows'][sample]);ix=a['neighbors'][sample]
    np.testing.assert_allclose(np.take_along_axis(sim,ix,axis=1),a['similarity'][sample],atol=2e-7)
    threshold=np.partition(sim,-ix.shape[1],axis=1)[:,-ix.shape[1]]
    assert np.all(np.take_along_axis(sim,ix,axis=1).min(1)>=threshold-3e-7)
    assert not np.any(d['canonical'][a['rows'][:,None]]==d['canonical'][fit][a['neighbors']])
    model=xgb.Booster();model.load_model(folder/'model.ubj')
    p=pair.predict_model(model,a,cfg['step']);pp=pair.predict_model(model,a,cfg['step'],batch=157)
    if r['task']=='ppbr_az':p=p.clip(0,100);pp=pp.clip(0,100)
    np.testing.assert_array_equal(p,z['values'][pair.SNAPSHOTS.index(cfg['step'])]);np.testing.assert_array_equal(p,pp)
    rec=dict(task=r['task'],part=part,status='PASS',model_sha256=meta['model_sha256'],descriptor_and_projection_reconstruction=True,
        independent_cpu_neighbor_samples=len(sample),query_label_perturbation=part!='final',subset_prediction=True,max_prediction_difference=0.)
    pair.emit(stage='independent_replay',**rec);return rec

def bsha(p):return count.b.sha(p)

def audit_boost(d,r,part):
    k=d['tasks'].index(r['task']);cfg=r['config'];boosted.expert(d,k,part,cfg);a=boosted.prepare(d,k,part)
    # Rebuild selected substructure columns directly from fit occurrence counts.
    for j,raw in enumerate(d['unfolded_fingerprints']):
        tr=raw[d['mi'][a['fit']]];expected=np.flatnonzero(np.asarray((tr>0).sum(0)).ravel()>=3)
        np.testing.assert_array_equal(a['columns'][j],expected)
    q=a['query']
    if part!='final':
        assert not set(d['groups'][a['fit']])&set(d['groups'][q]);saved=d['y'][q].copy();d['y'][q]=1e6;z=boosted.prepare(d,k,part);d['y'][q]=saved
        for key in ('fit','query','target','qoff'):np.testing.assert_array_equal(a[key],z[key])
        for key in ('x','qx'):
            for attr in ('data','indices','indptr'):np.testing.assert_array_equal(getattr(a[key],attr),getattr(z[key],attr))
    records=[];cls=r['metric'] in ('AUROC','AUPRC')
    for seed in boosted.SEEDS:
        folder=boosted.OUT/'models'/r['task']/part/f"depth{cfg['depth']}"/str(seed);meta=read(folder/'metadata.json');cache=dict(np.load(folder/'predictions.npz'))
        assert meta['signature']==str(cache['signature'])==boosted.sig()
        assert bsha(folder/'model.ubj')==meta['model_sha256'] and bsha(folder/'predictions.npz')==meta['prediction_sha256']
        for j in range(2):np.testing.assert_array_equal(cache[f'column{j}'],a['columns'][j])
        model=xgb.Booster();model.load_model(folder/'model.ubj');model.set_param({'device':'cpu','nthread':2})
        p=model.predict(xgb.DMatrix(a['qx'],nthread=2),iteration_range=(0,cfg['step']),output_margin=cls).astype(float)
        ids=np.unique([0,len(q)//2,len(q)-1]);sub=model.predict(xgb.DMatrix(a['qx'][ids],nthread=2),iteration_range=(0,cfg['step']),output_margin=cls).astype(float)
        np.testing.assert_array_equal(p[ids],sub)
        if not cls:p=p*float(a['scale'])+float(a['center'])
        if r['task']=='ppbr_az':p=p.clip(0,100)
        expected=cache['values'][boosted.SNAPSHOTS.index(cfg['step'])];np.testing.assert_allclose(p,expected,atol=2e-5,rtol=2e-5)
        records.append(dict(seed=seed,sha256=meta['model_sha256'],cpu_max_difference=float(np.max(np.abs(p-expected)))))
    rec=dict(task=r['task'],part=part,status='PASS',records=records,query_label_perturbation=part!='final',fit_only_columns=True,subset_prediction=True)
    boosted.emit(stage='independent_replay',**rec);return rec

def verify_branch(version):
    v=BRANCHES[version];d=v.load();rs=selected(v);records=[]
    choices=rs if rs else read(v.OUT/'frozen_selection.json')['selected']
    for r in choices:
        for part in (('fold0','fold1','fold2','valid','final') if rs else ('fold0','fold1','fold2','valid')):
            records.append(audit_pair(d,r,part) if version==103 else audit_boost(d,r,part) if version==106 else audit_margin(v,d,r,part))
    report=dict(status='PASS',version=version,accepted=[r['task'] for r in rs],records=records,no_csv=not bool(rs),test_labels_used=False)
    v.b.save(v.OUT/'model_verification.json',report);v.emit(stage='verification',version=version,accepted=report['accepted'],status='PASS')

def assemble():
    d=count.load();chosen={};deps=[];protocol=read(ROOT/'docs/V107_ASSEMBLY_PROTOCOL_20261026.json')
    assert protocol['branches']==list(BRANCHES) and protocol['anchor_sha256']==count.ANCHOR_SHA
    for version,v in BRANCHES.items():
        audit=read(v.OUT/'model_verification.json');assert audit['status']=='PASS'
        deps.append(dict(version=version,signature=v.sig(),confirmation_sha256=bsha(v.OUT/'confirmation.json'),verification_sha256=bsha(v.OUT/'model_verification.json')))
        for r in selected(v):
            if r['task'] not in chosen or r['objective']>chosen[r['task']]['record']['objective']:chosen[r['task']]=dict(version=version,record=r)
    assert chosen,'No accepted improvements; do not reissue v101 under a new name.'
    old=table(count.ANCHOR);rows=[r.copy() for r in old];changed={};checks=[]
    for task,ch in chosen.items():
        v=BRANCHES[ch['version']];dd=v.load() if v in (unfolded,boosted) else d;r=ch['record'];k=d['tasks'].index(task)
        for part in ('fold0','fold1','fold2','valid','final'):
            p=v.mix(r['metric'],v.base(dd,k,part),v.expert(dd,k,part,r['config']),r['weight'])
            if part=='final':
                ids=[i for i,row in enumerate(old) if i and row[0]==task];assert len(p)==len(ids) and np.isfinite(p).all()
                for i,x in zip(ids,p):rows[i][4]=repr(float(x))
                changed[task]=sum(rows[i][4]!=old[i][4] for i in ids)
            else:
                q=count.r83.split(d,k,part)[1];s=count.r83.score(r['metric'],d['y'][q],p);ref=next(x for x in r['confirmation'] if x['part']==part)
                np.testing.assert_allclose(s,ref['candidate'],atol=1e-12,rtol=1e-12);checks.append(dict(task=task,version=ch['version'],part=part,score=s,gain=ref['gain']))
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();sp=field_spans(raw);chunks=[];cursor=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=sp[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks)
    file=ROOT/'admet_submissions/v107_count_difference_boost/我从山北来带着兰花草-小分子药物预测技术创新赛-20261026.csv';file.parent.mkdir(parents=True,exist_ok=True)
    if file.exists():assert file.read_bytes()==output
    else:file.write_bytes(output)
    actual=table(file);observed=Counter()
    for a,z in zip(old[1:],actual[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[4]!=z[4]:observed[z[0]]+=1
        if z[0] not in chosen:assert a==z
    assert dict(observed)==changed;ss=field_spans(output);back=[];cursor=0
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];a,z=sp[i][4];back.extend((output[cursor:lo],raw[a:z]));cursor=hi
    back.append(output[cursor:]);assert b''.join(back)==raw
    fmt=verify(file,template);fmt.update(nonprediction_bytes_exact=True,model_replay='PASS',changed_tasks=changed)
    count.b.save(file.with_suffix('.verification.json'),fmt)
    report=dict(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',version='v107',baseline='v101',anchor_sha256=count.ANCHOR_SHA,
        baseline_official_score_user_reported=50.932,file=file.relative_to(ROOT).as_posix(),sha256=bsha(file),selected=chosen,changed_tasks=changed,
        unchanged_tasks=22-len(changed),verification=fmt,checks=checks,dependencies=deps,protocol_sha256=bsha(ROOT/'docs/V107_ASSEMBLY_PROTOCOL_20261026.json'),
        official_score=None,test_labels_used=False,new_external_data=False,new_pretrained_weights=False,automatically_uploaded=False)
    count.b.save(OUT/'release.json',report);count.b.save(OUT/'model_verification.json',dict(status='PASS',branches=deps,checks=checks,submission=fmt))
    count.emit(stage='assembly',status='PASS',changed_tasks=changed,sha256=report['sha256'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['verify','assemble']);ap.add_argument('--version',type=int,choices=list(BRANCHES));args=ap.parse_args()
    count.torch.set_num_threads(2);count.torch.backends.cuda.matmul.allow_tf32=False;count.torch.set_float32_matmul_precision('highest')
    if args.stage=='verify':verify_branch(args.version)
    else:assemble()
