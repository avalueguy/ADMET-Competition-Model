"""Fit-only neighbor-referenced property differences for regression endpoints.

Learn antisymmetrized changes between labeled fit molecules, then estimate a
query by adding predicted changes to fit references. No query labels in pairs.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch,xgboost as xgb
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v102_count_margin as prev
b,r83=prev.b,prev.r83
OUT=ROOT/'admet_model_outputs_v103';ANCHOR,ANCHOR_SHA=prev.ANCHOR,prev.ANCHOR_SHA
DEPTHS=(3,6);STEPS=700;SNAPSHOTS=(200,700);SEED=1034026;NEIGHBORS=16;WEIGHTS=(.25,.5,1.)

def sig():return hashlib.sha256((b.sha(__file__)+prev.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def load():return prev.load()
def base(d,k,part):return prev.base(d,k,part)
def mix(m,old,p,w):return prev.mix(m,old,p,w)
def threshold(m):return prev.threshold(m)
def configs(m):return [dict(depth=z,step=t) for z in DEPTHS for t in SNAPSHOTS]

def projection(d):
    if '_v103_projection' not in d:
        raw=d['views']['multiview'][:,:6144];x=sparse.csr_matrix(np.rint(np.expm1(raw)))
        x.data=np.sqrt(x.data)
        w=np.random.default_rng(SEED).normal(0,1/8,(6144,64)).astype(np.float32)
        d['_v103_projection']=(x@w).astype(np.float32)
    return d['_v103_projection']

def nearest(d,rows,fit,top=NEIGHBORS):
    ref=d['mi'][fit];query=d['mi'][rows];sim=torch.zeros((len(rows),len(fit)),device='cuda')
    for view in range(3):
        tr=torch.as_tensor(prev.count_raw(d,ref,view),device='cuda');tn=tr.square().sum(1)
        for lo in range(0,len(query),256):
            x=torch.as_tensor(prev.count_raw(d,query[lo:lo+256],view),device='cuda');dot=x@tr.T
            val=(dot/(x.square().sum(1)[:,None]+tn[None]-dot).clamp_min(1e-8)).clamp(0,1)
            sim[lo:lo+len(x)]+=val/3
    for lo in range(0,len(rows),256):
        exact=d['canonical'][rows[lo:lo+256],None]==d['canonical'][fit][None]
        sim[lo:lo+len(exact)].masked_fill_(torch.as_tensor(exact,device='cuda'),-1.)
    values,ix=torch.topk(sim,min(top,len(fit)-1),dim=1,sorted=True)
    values,ix=values.cpu().numpy(),ix.cpu().numpy();assert np.min(values)>=0
    return ix,values

def prepare(d,k,part):
    fit,q=r83.split(d,k,part);metric=d['metrics'][k];assert metric in ('MAE','Spearman')
    rows=np.r_[fit,q];fi=d['mi'][fit];mi=d['mi'][rows]
    tx,qx,center,scale,active=prev.kernel.normalized(d['views']['physchem'],fi,mi)
    # This normalization only sees fit labels; all query labels are ignored.
    y=d['y'][fit];yc=float(np.median(y)) if metric=='MAE' else 0.
    ys=max(float(np.quantile(y,.75)-np.quantile(y,.25)),.001) if metric=='MAE' else 1.
    target=(y-yc)/ys if metric=='MAE' else r83.ranknorm(y)
    nn,sim=nearest(d,rows,fit);rp=projection(d)
    rcenter=rp[fi].mean(0);rscale=rp[fi].std(0).clip(.05)
    projected=((rp[mi]-rcenter)/rscale).clip(-6,6).astype(np.float32)
    features=np.concatenate((qx,projected),axis=1)
    return dict(fit=fit,query=q,rows=rows,features=features,neighbors=nn,similarity=sim,target=target,
        center=np.array(yc),scale=np.array(ys),descriptor_center=center,descriptor_scale=scale,active=active,
        projection_center=rcenter,projection_scale=rscale)

def pair_features(a,left,right):
    x=a['features'][left];z=a['features'][right];delta=x-z
    return np.concatenate((delta,np.abs(delta),.5*(x+z)),axis=1).astype(np.float32)

def pair_data(a):
    n=len(a['fit']);left=np.repeat(np.arange(n),a['neighbors'].shape[1]);right=a['neighbors'][:n].ravel()
    weight=np.maximum(a['similarity'][:n].ravel(),.01)**2
    return np.r_[left,right],np.r_[right,left],np.r_[a['target'][left]-a['target'][right],a['target'][right]-a['target'][left]],np.r_[weight,weight]

def predict_model(model,a,step,batch=2048):
    n=len(a['fit']);left=np.repeat(np.arange(n,len(a['rows'])),a['neighbors'].shape[1]);right=a['neighbors'][n:].ravel();values=[]
    for lo in range(0,len(left),batch):
        l,r=left[lo:lo+batch],right[lo:lo+batch]
        p=model.predict(xgb.DMatrix(pair_features(a,l,r)),iteration_range=(0,step))
        z=model.predict(xgb.DMatrix(pair_features(a,r,l)),iteration_range=(0,step))
        values.append(a['target'][r]+.5*(p-z))
    anchored=np.concatenate(values).reshape(len(a['query']),-1);weights=np.maximum(a['similarity'][n:],.001)**4
    # Fixed weighted median is robust to inconsistent neighboring measurements.
    order=np.argsort(anchored,axis=1);vals=np.take_along_axis(anchored,order,axis=1);ww=np.take_along_axis(weights,order,axis=1)
    pos=(np.cumsum(ww,axis=1)<.5*ww.sum(1)[:,None]).sum(1).clip(0,ww.shape[1]-1)
    return vals[np.arange(len(vals)),pos]*float(a['scale'])+float(a['center'])

def train(d,k,part,depth):
    folder=OUT/'models'/d['tasks'][k]/part/f'depth{depth}';file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);meta=json.loads((folder/'metadata.json').read_text());assert str(z['signature'])==meta['signature']==sig()
        assert b.sha(file)==meta['prediction_sha256'] and b.sha(folder/'model.ubj')==meta['model_sha256']
        np.testing.assert_array_equal(z['query'],r83.split(d,k,part)[1]);return z['values']
    a=prepare(d,k,part);left,right,target,weights=pair_data(a);started=time.monotonic()
    dm=xgb.DMatrix(pair_features(a,left,right),label=target,weight=weights,nthread=6)
    model=xgb.train(dict(objective='reg:absoluteerror' if d['metrics'][k]=='MAE' else 'reg:squarederror',tree_method='hist',device='cpu',
        max_depth=depth,max_bin=64,eta=.035,min_child_weight=10,subsample=.85,colsample_bytree=.8,reg_lambda=20,reg_alpha=.05,
        seed=SEED,nthread=6),dm,num_boost_round=STEPS)
    values=np.stack([predict_model(model,a,t) for t in SNAPSHOTS])
    if d['tasks'][k]=='ppbr_az':values=values.clip(0,100)
    folder.mkdir(parents=True,exist_ok=True);model.save_model(folder/'model.ubj')
    np.savez_compressed(file,signature=sig(),fit=a['fit'],query=a['query'],values=values,snapshots=SNAPSHOTS)
    state={key:val for key,val in a.items() if key not in ('features',)};np.savez_compressed(folder/'input_state.npz',signature=sig(),**state)
    b.save(folder/'metadata.json',dict(signature=sig(),task=d['tasks'][k],part=part,depth=depth,steps=STEPS,seed=SEED,pairs=len(left),
        model_sha256=b.sha(folder/'model.ubj'),prediction_sha256=b.sha(file),input_state_sha256=b.sha(folder/'input_state.npz'),seconds=time.monotonic()-started,
        target_fit_only=True,query_labels_used=False));emit(stage='train',task=d['tasks'][k],part=part,depth=depth,seconds=round(time.monotonic()-started));return values

def expert(d,k,part,cfg):return train(d,k,part,cfg['depth'])[SNAPSHOTS.index(cfg['step'])]

def protocol():
    OUT.mkdir(exist_ok=True);doc=dict(version=103,signature=sig(),baseline='v101',baseline_official_score_user_reported=50.932,anchor_sha256=ANCHOR_SHA,
        tasks='All MAE and Spearman tasks',depths=list(DEPTHS),steps=STEPS,snapshots=list(SNAPSHOTS),seed=SEED,neighbors=NEIGHBORS,weights=list(WEIGHTS),
        pair_input='Fit-only standardized 435 descriptor columns plus deterministic 64D count-fingerprint projection. Delta, absolute delta, midpoint.',
        pair_target='Fit reference normalized property differences. Symmetric pairs; own fit labels only.',
        query_prediction='Antisymmetrize pair model; add known fit reference property; top16 count-Tanimoto references excluding exact identity; similarity^4 weighted median.',
        selection='Max minimum gain on fold0/1 >1% relative MAE or .005 Spearman; freeze one configuration before confirmation.',
        confirmation='All four gains positive and average above threshold, no failed fallback.',test_labels_used=False,new_external_data=False,new_pretrained_weights=False,
        limitations='Previously reused historical labels; proxy baseline is not exact v101 OOF; no official score estimate.')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==doc
    else:b.save(path,doc)

def develop(d):
    protocol();records=[];selected=[]
    tasks=[k for k,m in enumerate(d['metrics']) if m in ('MAE','Spearman')]
    tasks.sort(key=lambda k:int(((d['tid']==k)&(d['split']==0)).sum()))
    for k in tasks:
        m=d['metrics'][k];bases={p:base(d,k,p) for p in ('fold0','fold1')}
        for cfg in configs(m):
            pp={p:expert(d,k,p,cfg) for p in bases}
            for w in WEIGHTS:
                parts=[]
                for part,old in bases.items():
                    q=r83.split(d,k,part)[1];p=mix(m,old,pp[part],w)
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
        rec=dict(**ch,confirmation=parts,accepted=bool(all(x['gain']>0 for x in parts) and np.mean([x['gain'] for x in parts])>threshold(m)))
        records.append(rec);emit(stage='confirmation',**rec);b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))

def smoke(d):
    protocol();k=d['tasks'].index('half_life_obach');a=prepare(d,k,'fold0');q=a['query'];saved=d['y'][q].copy();d['y'][q]=1e6;z=prepare(d,k,'fold0');d['y'][q]=saved
    for key in a:np.testing.assert_array_equal(a[key],z[key])
    n=len(a['fit']);left,right,yt,w=pair_data(a);assert max(left.max(),right.max())<n
    assert not np.any(d['canonical'][a['rows'][left]]==d['canonical'][a['rows'][right]])
    np.testing.assert_array_equal(pair_features(a,left[:20],right[:20])[:,:a['features'].shape[1]],-pair_features(a,right[:20],left[:20])[:,:a['features'].shape[1]])
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),query_label_perturbation=True,fit_only_reference_labels=True,exact_identity_excluded=True,antisymmetry=True));emit(stage='smoke',status='PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['protocol','smoke','develop','confirm']);args=ap.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    if args.stage=='protocol':protocol()
    else:dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](load())
