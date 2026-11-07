"""Count-sensitive molecular margin experts over the official v101 anchor.

Target fit labels only. Raw substructure counts reconstructed from versioned
log-count features. Own-task validation/query labels never enter training.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import joblib,numpy as np,torch
from sklearn.svm import SVC,SVR
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v95_graph_specialists as graph
import v96_normalized_assays as assay
import v98_shape_experts as shape
import v100_shape_kernel as kernel
b,r83=graph.b,graph.r83
OUT=ROOT/'admet_model_outputs_v102'
ANCHOR=ROOT/'admet_submissions/v101_graph_assay_shape/我从山北来带着兰花草-小分子药物预测技术创新赛-20261026.csv'
ANCHOR_SHA='e94961c81deedf98bcb69e04d621a54ae9319aab47bb1ced9d97890f13c8f086'
CS=(.3,3.,30.);VIEWS=('count','circular','hybrid');WEIGHTS=(.25,.5,1.)

def sig():return hashlib.sha256((b.sha(__file__)+graph.sig()+assay.sig()+shape.sig()+kernel.sig()+ANCHOR_SHA).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def load():
    d=shape.load();assert b.sha(ANCHOR)==ANCHOR_SHA
    release=json.loads((ROOT/'admet_model_outputs_v101/release.json').read_text())
    assert release['sha256']==ANCHOR_SHA and release['verification']['nonprediction_bytes_exact']
    d['v101_selected']=release['selected'];return d

def base(d,k,part):
    if part=='final':return np.array([float(r['y_pred']) for r in b.table(ANCHOR) if r['task']==d['tasks'][k]])
    old=graph.base(d,k,part);chosen=d['v101_selected'].get(d['tasks'][k])
    if chosen:
        v={'v95':graph,'v96':assay,'v98':shape,'v100':kernel}[chosen['version']];r=chosen['record']
        old=v.mix(r['metric'],old,v.expert(d,k,part,r['config']),r['weight'])
    return old

def mix(m,a,z,w):return graph.mix(m,a,z,w)
def threshold(m):return .01 if m=='MAE' else .005 if m=='Spearman' else .002
def configs(metric):
    modes=('direct',) if metric in ('AUROC','AUPRC') else ('direct','residual')
    return [dict(view=v,mode=m,C=c) for v in VIEWS for m in modes for c in CS]

def state(d,k,part):
    a=kernel.historical.state(d,k,part)
    return a

def count_raw(d,mols,view):
    log=d['views']['multiview'][mols,view*2048:(view+1)*2048]
    return np.rint(np.expm1(log)).astype(np.float32)

def matrices(d,k,part,small=False):
    fit,q=r83.split(d,k,part)
    if small:fit,q=fit[:70],q[:13]
    fi,qi=d['mi'][fit],d['mi'][q];n=len(fit)
    full=torch.zeros((n,n),device='cuda');fq=torch.zeros((len(q),n),device='cuda')
    circular=torch.zeros_like(full);cq=torch.zeros_like(fq)
    for view in range(6):
        raw=count_raw(d,fi,view);tr=torch.as_tensor(raw,device='cuda');tn=tr.square().sum(1)
        for ids,dest,cdest in ((fi,full,circular),(qi,fq,cq)):
            for lo in range(0,len(ids),256):
                x=torch.as_tensor(count_raw(d,ids[lo:lo+256],view),device='cuda');dot=x@tr.T
                sim=(dot/(x.square().sum(1)[:,None]+tn[None]-dot).clamp_min(1e-8)).clamp(0,1)
                val=.5*sim+.5*sim.square();dest[lo:lo+len(x)]+=val/6
                if view in (0,1,2):cdest[lo:lo+len(x)]+=val/3
    tx,qx,center,scale,active=kernel.normalized(d['views']['physchem'],fi,qi)
    tr=torch.as_tensor(tx,device='cuda');tn=tr.square().sum(1);pr=torch.zeros_like(full);pq=torch.zeros_like(fq)
    for data,dest in ((tx,pr),(qx,pq)):
        for lo in range(0,len(data),256):
            x=torch.as_tensor(data[lo:lo+256],device='cuda');dist=(x.square().sum(1)[:,None]+tn[None]-2*x@tr.T).clamp_min(0)/max(1,len(active))
            dest[lo:lo+len(x)]=.5*torch.exp(-.5*dist)+.5*torch.exp(-2*dist)
    tr={'count':full,'circular':circular,'hybrid':.8*full+.2*pr}
    qe={'count':fq,'circular':cq,'hybrid':.8*fq+.2*pq}
    result={v:((tr[v]+tr[v].T).mul(.5).cpu().double().numpy(),qe[v].cpu().double().numpy()) for v in VIEWS}
    del full,fq,circular,cq,tr,qe,pr,pq;torch.cuda.empty_cache()
    return result,dict(center=center,scale=scale,active=active)

def train_one(d,k,part,cfg,a,matrices):
    task=d['tasks'][k];name=f"{cfg['view']}_{cfg['mode']}_c{cfg['C']}"
    folder=OUT/'models'/task/part/name;file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);meta=json.loads((folder/'metadata.json').read_text())
        assert str(z['signature'])==meta['signature']==sig() and b.sha(file)==meta['predictions_sha256']
        assert b.sha(folder/'model.joblib')==meta['model_sha256']
        for key in ('fit','query'):np.testing.assert_array_equal(z[key],a[key])
        return z['p'].copy()
    started=time.monotonic();metric=d['metrics'][k];cls=metric in ('AUROC','AUPRC');x,qx=matrices[cfg['view']]
    model=SVC(C=cfg['C'],kernel='precomputed',tol=1e-4,cache_size=768,probability=False,shrinking=True) if cls else SVR(C=cfg['C'],epsilon=.03,kernel='precomputed',tol=1e-4,cache_size=768,shrinking=True)
    yt=d['y'][a['fit']] if cls else a['target'][:,('direct','residual').index(cfg['mode'])]
    model.fit(x,yt);assert model.fit_status_==0
    raw=model.decision_function(qx) if cls else model.predict(qx)
    if cls:p=expit(raw)
    elif cfg['mode']=='direct':p=raw*float(a['scale'])+float(a['center'])
    else:p=raw*float(a['scale'])+a['qoff']
    if task=='ppbr_az':p=p.clip(0,100)
    assert np.isfinite(p).all();folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,folder/'model.joblib',compress=3)
    np.savez_compressed(file,signature=sig(),**a,p=p,raw=raw)
    b.save(folder/'metadata.json',dict(signature=sig(),task=task,part=part,config=cfg,model_sha256=b.sha(folder/'model.joblib'),
        predictions_sha256=b.sha(file),seconds=time.monotonic()-started,support_vectors=len(model.support_),fit_status=int(model.fit_status_),
        own_task_fit_labels_only=True,new_external_data=False,query_labels_in_training=False))
    emit(stage='train',task=task,part=part,config=cfg,seconds=round(time.monotonic()-started));return p

def family(d,k,part,candidates=None):
    cfgs=configs(d['metrics'][k]) if candidates is None else candidates
    memo=d.setdefault('_v102_models',{});missing=[c for c in cfgs if (k,part,json.dumps(c,sort_keys=True)) not in memo]
    if missing:
        a=state(d,k,part);mm,stats=matrices(d,k,part)
        folder=OUT/'models'/d['tasks'][k]/part;folder.mkdir(parents=True,exist_ok=True)
        statfile=folder/'descriptor_stats.npz'
        if not statfile.exists():np.savez_compressed(statfile,signature=sig(),**stats)
        with ThreadPoolExecutor(max_workers=2) as pool:
            ps=list(pool.map(lambda c:train_one(d,k,part,c,a,mm),missing))
        for c,p in zip(missing,ps):memo[(k,part,json.dumps(c,sort_keys=True))]=p
    return {json.dumps(c,sort_keys=True):memo[(k,part,json.dumps(c,sort_keys=True))] for c in cfgs}

def expert(d,k,part,cfg):return family(d,k,part,[cfg])[json.dumps(cfg,sort_keys=True)]

def protocol():
    OUT.mkdir(exist_ok=True);doc=dict(version=102,signature=sig(),baseline='v101',anchor_sha256=ANCHOR_SHA,baseline_official_score_user_reported=50.932,
        C=list(CS),views=list(VIEWS),weights=list(WEIGHTS),classification='SVC hinge margin, own-task labels, no probability calibration.',
        regression='SVR epsilon .03 on IQR-normalized MAE or Gaussian rank target; direct and historical v73 residual targets.',
        counts='Round expm1 of cached log counts. Original first two fingerprint channels had clipping at log6; retain this documented truncation.',
        kernels='Generalized count Tanimoto .5 T+.5 T^2, six equally weighted views; circular first three; hybrid .8count+.2 fit-only 2D RBF.',
        selection='Max minimum gain fold0/1, >1% MAE/.005 Spearman/.002 AUC/AP. One frozen configuration per task.',
        confirmation='Four positive gains and average above practical threshold. No fallback after failure.',
        test_labels_used=False,new_external_data=False,new_pretrained_weights=False,
        limitations='Historical proxy is approximate, not complete v101 OOF; previously reused labels; no official score estimate.')
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==doc
    else:b.save(p,doc)
    return doc

def develop(d):
    protocol();records=[];selected=[]
    order=sorted(range(len(d['tasks'])),key=lambda k:int(((d['tid']==k)&(d['split']==0)).sum()))
    for k in order:
        m=d['metrics'][k];bases={p:base(d,k,p) for p in ('fold0','fold1')};ps={p:family(d,k,p) for p in bases}
        for cfg in configs(m):
            for w in WEIGHTS:
                parts=[]
                for part,old in bases.items():
                    q=r83.split(d,k,part)[1];p=mix(m,old,ps[part][json.dumps(cfg,sort_keys=True)],w)
                    parts.append(dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],old),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],old,p)))
                records.append(dict(task=d['tasks'][k],metric=m,config=cfg,weight=w,parts=parts,objective=min(x['gain'] for x in parts)))
        best=max((r for r in records if r['task']==d['tasks'][k]),key=lambda r:r['objective']);emit(stage='development',**best)
        if best['objective']>threshold(m):selected.append(best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records))
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected,confirmation_inspected=False));emit(stage='frozen',selected=selected)

def confirm(d):
    f=json.loads((OUT/'frozen_selection.json').read_text());assert f['signature']==sig();records=[]
    for ch in f['selected']:
        k=d['tasks'].index(ch['task']);m=ch['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];old=base(d,k,part);p=mix(m,old,expert(d,k,part,ch['config']),ch['weight'])
            parts.append(dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],old),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],old,p)))
        record=dict(**ch,confirmation=parts,accepted=bool(all(x['gain']>0 for x in parts) and np.mean([x['gain'] for x in parts])>threshold(m)))
        records.append(record);emit(stage='confirmation',**record)
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))

def smoke(d):
    protocol();k=d['tasks'].index('cyp2c9_substrate_carbonmangels');mm,_=matrices(d,k,'fold0',True)
    for name,(tr,q) in mm.items():
        np.testing.assert_allclose(tr,tr.T,atol=1e-10);assert np.linalg.eigvalsh(tr).min()>-1e-5
        model=SVC(C=3.,kernel='precomputed').fit(tr,d['y'][r83.split(d,k,'fold0')[0][:70]])
        p=model.decision_function(q);np.testing.assert_allclose(p,q[:,model.support_]@model.dual_coef_[0]+model.intercept_[0],atol=1e-10)
    a=state(d,k,'fold0');q=a['query'];saved=d['y'][q].copy();d['y'][q]=1e6
    z=state(d,k,'fold0');d['y'][q]=saved
    for key in a:np.testing.assert_array_equal(a[key],z[key])
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),kernels_psd=True,manual_support_vector_replay=True,query_label_perturbation=True));emit(stage='smoke',status='PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','develop','confirm','protocol']);args=ap.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    if args.stage=='protocol':protocol()
    else:dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](load())
