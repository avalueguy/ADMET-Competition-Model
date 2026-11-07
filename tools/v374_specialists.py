"""Remote-only MAE experts: alternative similarity, replicate weighting, honest residuals."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'4')
import hashlib,json,time
import numpy as np,torch,joblib
from scipy.linalg import cho_factor,cho_solve
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import v371_mae_experts as shared

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
TASKS=shared.TASKS
WEIGHTS=(.1,.2,.45,.75,1.)
CONFIGS=tuple([dict(name=f'{v}_C{int(c)}',family='kernel',view=v,C=c,weighted=False) for v in ('raw_count','raw_circular','binary_circular') for c in (3.,30.)]
    +[dict(name=f'replicate_weighted_C{int(c)}',family='kernel',view='raw_count',C=c,weighted=True) for c in (3.,30.)]
    +[dict(name=f'honest_local_k{k}',family='residual',neighbors=k,tree=False) for k in (16,64)]
    +[dict(name=f'honest_residual_tree_d{k}',family='residual',depth=k,tree=True) for k in (2,4)])
sha=shared.sha
save=shared.save
emit=shared.emit
target=shared.target
clip=shared.clip

def setup():shared.setup()
def signature():return hashlib.sha256((sha(__file__)+sha(shared.__file__)+sha(ROOT/'data_manifest.json')+json.dumps([CONFIGS,WEIGHTS],sort_keys=True)).encode()).hexdigest()
def load():
    shared.ROOT=ROOT;d=shared.load();d.update(dict(np.load(ROOT/'stereo_keys.npz')));return d
def ctrl(task,part):return shared.ctrl(task,part)
def design(d,task,part):return shared.design(d,task,part)

def matrices(d,a,task,part,tag='full'):
    path=OUT/'kernels'/task/part/(tag+'.npz')
    if path.exists():
        z=np.load(path);assert str(z['signature'])==signature()
        for key in ('fit','query'):np.testing.assert_array_equal(z[key],a[key])
        return {v:(z[v+'_train'],z[v+'_query']) for v in ('raw_count','raw_circular','binary_circular')}
    n,q=len(a['fit']),len(a['query']);raw=np.zeros((n,n),np.float32);rq=np.zeros((q,n),np.float32)
    circ=np.zeros_like(raw);cq=np.zeros_like(rq);binary=np.zeros_like(raw);bq=np.zeros_like(rq)
    fi,qi=d['mi'][a['fit']],d['mi'][a['query']]
    for view in range(6):
        # Cache contains log1p counts. Restore the historical capped counts explicitly.
        values=np.rint(np.expm1(d['fingerprints'][:,view*2048:(view+1)*2048])).astype(np.float32)
        f=torch.as_tensor(values[fi],device='cuda');fn=f.square().sum(1)
        for ids,o,co in ((fi,raw,circ),(qi,rq,cq)):
            for lo in range(0,len(ids),256):
                x=torch.as_tensor(values[ids[lo:lo+256]],device='cuda');dot=x@f.T
                t=(dot/(x.square().sum(1)[:,None]+fn[None]-dot).clamp_min(1e-8)).clamp(0,1)
                val=(.5*t+.5*t.square()).cpu().numpy();o[lo:lo+len(x)]+=val/6
                if view<3:co[lo:lo+len(x)]+=val/3
        if view<3:
            f=(f>0).float();fn=f.sum(1)
            for ids,o in ((fi,binary),(qi,bq)):
                for lo in range(0,len(ids),256):
                    x=torch.as_tensor(values[ids[lo:lo+256]]>0,dtype=torch.float32,device='cuda');dot=x@f.T
                    t=(dot/(x.sum(1)[:,None]+fn[None]-dot).clamp_min(1e-8)).clamp(0,1)
                    o[lo:lo+len(x)]+=(.5*t+.5*t.square()).cpu().numpy()/3
    kr,kq,_=shared.rbf(a['x'],a['qx']);pr,pq,_=shared.rbf(d['physical'][fi],d['physical'][qi])
    result={v:((.8*k+.15*kr+.05*pr),(.8*z+.15*kq+.05*pq)) for v,k,z in (('raw_count',raw,rq),('raw_circular',circ,cq),('binary_circular',binary,bq))}
    for v,(k,z) in result.items():result[v]=((k+k.T)*.5,z)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,signature=signature(),fit=a['fit'],query=a['query'],**{v+'_'+side:value for v,(k,z) in result.items() for side,value in (('train',k),('query',z))})
    return result

def replicate_weights(d,fit):
    keys=d['isomeric'][d['mi'][fit]];unique,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
    y=d['y'][fit];_,_,scale=target(y);w=np.ones(len(fit));records=[]
    for i in range(len(unique)):
        ix=np.flatnonzero(inverse==i);spread=float(np.median(np.abs(y[ix]-np.median(y[ix]))))
        weight=1/np.sqrt(len(ix))/(1+(spread/max(.5*scale,.01))**2)
        w[ix]=weight
        if len(ix)>1:records.append(dict(count=len(ix),range=float(np.ptp(y[ix])),mad=spread,weight=weight))
    w=np.clip(w/w.mean(),.25,2.)
    return w,dict(groups=len(unique),replicate_groups=len(records),rows_in_replicates=int(sum(r['count'] for r in records)),replicates=records)

def ridge(d,a,task,part,tag):
    k,kq=matrices(d,a,task,part,tag)['raw_count'];yy,c,s=target(d['y'][a['fit']]);coef=cho_solve(cho_factor(k.astype(float)+np.eye(len(k))*.3),yy)
    return kq@coef*s+c,dict(coef=coef,center=c,scale=s)

def residual_design(d,a,ref,query,sim):
    x,qx,state=shared.robust(a['x'],a['qx'])
    # Inputs at fit are honest OOF base estimates. Query gets fullfit estimate.
    _,c,s=target(d['y'][a['fit']]);ix=sim.argsort(axis=1)[:,-8:]
    top=np.take_along_axis(sim,ix,1)
    return np.c_[x,(ref-c)/s],np.c_[qx,(query-c)/s],state

def residual_state(d,a,task,part):
    path=OUT/'residuals'/task/part/'state.joblib'
    if path.exists():
        s=joblib.load(path);assert s['signature']==signature();return s
    fit=a['fit'];oof=np.full(len(fit),np.nan);splits=[]
    for i,(tr,ho) in enumerate(GroupKFold(3).split(fit,groups=d['rg'][fit])):
        sub=dict(fit=fit[tr],query=fit[ho],x=a['x'][tr],qx=a['x'][ho],source=a['source'])
        p,state=ridge(d,sub,task,part,'inner'+str(i));oof[ho]=p
        splits.append(dict(train=tr,held=ho,state=state))
    base,state=ridge(d,a,task,part,'full');k,kq=matrices(d,a,task,part)['raw_count']
    assert np.isfinite(oof).all()
    weights,audit=replicate_weights(d,fit)
    s=dict(signature=signature(),fit=fit,query=a['query'],oof=oof,residual=d['y'][fit]-oof,base=base,ridge_state=state,splits=splits,weights=weights,replicate_audit=audit)
    path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(s,path,compress=3)
    return s

def predict(d,task,part,c):
    setup();folder=OUT/'tasks'/task/part/c['name'];path=folder/'predictions.npz';sig=signature()
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig;return z['p']
    start=time.monotonic();a=design(d,task,part);fit=a['fit'];folder.mkdir(parents=True,exist_ok=True)
    state=dict(signature=sig,config=c,fit=fit,query=a['query'])
    if c['family']=='kernel':
        k,kq=matrices(d,a,task,part)[c['view']];y,center,scale=target(d['y'][fit]);w,audit=replicate_weights(d,fit)
        net=SVR(kernel='precomputed',C=c['C'],epsilon=.01,tol=1e-4,max_iter=250000,cache_size=1000).fit(k,y,sample_weight=w if c['weighted'] else None)
        assert net.fit_status_==0
        p=net.predict(kq)*scale+center;state.update(model=net,center=center,scale=scale,replicate_audit=audit,weights=w)
    else:
        rs=residual_state(d,a,task,part);k,kq=matrices(d,a,task,part)['raw_count']
        if c['tree']:
            x,qx,pre=residual_design(d,a,rs['oof'],rs['base'],kq)
            rr,c0,s0=target(rs['residual']);net=CatBoostRegressor(loss_function='MAE',iterations=450,depth=c['depth'],learning_rate=.025,l2_leaf_reg=40,
                random_seed=3741106,task_type='GPU',devices='0',thread_count=4,verbose=False,allow_writing_files=False,border_count=64,bootstrap_type='Bernoulli',subsample=.85)
            net.fit(x,rr,sample_weight=rs['weights']);correction=net.predict(qx)*s0+c0;p=rs['base']+.5*correction
            state.update(model=net,preprocess=pre,center=c0,scale=s0)
        else:
            n=min(c['neighbors'],len(fit));ix=np.argsort(-kq,axis=1,kind='stable')[:,:n];sim=np.take_along_axis(kq,ix,1)
            weights=np.maximum(sim,1e-6)**4*rs['weights'][ix]
            p=rs['base']+.5*shared.weighted_median(rs['residual'][ix],weights)
    p=clip(task,p);assert np.isfinite(p).all();joblib.dump(state,folder/'model.joblib',compress=3)
    np.savez_compressed(path,signature=sig,fit=fit,query=a['query'],p=p)
    save(folder/'metadata.json',dict(signature=sig,task=task,part=part,config=c,seconds=time.monotonic()-start,model_sha256=sha(folder/'model.joblib'),predictions_sha256=sha(path)))
    emit('fit_done',task=task,part=part,config=c['name'],seconds=time.monotonic()-start)
    return p

def replay(d,task,part,c):
    a=design(d,task,part);state=joblib.load(OUT/'tasks'/task/part/c['name']/'model.joblib');assert state['signature']==signature()
    for key in ('fit','query'):np.testing.assert_array_equal(state[key],a[key])
    if c['family']=='kernel':
        k,kq=matrices(d,a,task,part)[c['view']];p=state['model'].predict(kq)*state['scale']+state['center']
    else:
        rs=joblib.load(OUT/'residuals'/task/part/'state.joblib');assert rs['signature']==signature();_,kq=matrices(d,a,task,part)['raw_count']
        st=rs['ridge_state'];base=kq@st['coef']*st['scale']+st['center']
        if c['tree']:
            _,qx,_=residual_design(d,a,rs['oof'],base,kq);p=base+.5*(state['model'].predict(qx)*state['scale']+state['center'])
        else:
            n=min(c['neighbors'],len(a['fit']));ix=np.argsort(-kq,axis=1,kind='stable')[:,:n];w=np.maximum(np.take_along_axis(kq,ix,1),1e-6)**4*rs['weights'][ix]
            p=base+.5*shared.weighted_median(rs['residual'][ix],w)
    return clip(task,p)

def freeze():
    v=dict(signature=signature(),configs=CONFIGS,weights=WEIGHTS,tasks=TASKS,baseline='v371 local exactCSV; human official51.270, solubility provenance mismatch retained',
        methods='Raw historical capped counts reconstructed with expm1; circular raw and binary kernels; fit-only duplicate inverse-root count/MAD weights; three-inner-fold ridge residual median and GPU MAE trees.',
        inner_honesty='Each inner kernel preprocessing rebuilt on innerfit. Same-task query labels excluded, stereo replicate weights use fit only. No in-sample historical predictions as offsets.',
        context='Other official train observations, additional other valid for final; target and mask excluded.',all_fitting_remote=True,external_data=False,test_labels_used=False,
        selection='Unique maxmin fold0/1 configuration and blend frozen before fold2/valid. Failed confirmation has no fallback.',
        limitations='Historical v364 controls partly proxy, exact v371 LD50 layer replayed. Repeated public labels and concentrated scaffolds; no official prediction guarantee. Full historical ensemble OOF not yet rebuilt.',
        strict='Fourparts positive, devmin>1.2percent, mean>1.2percent and both500groupbootstrap90lower>0.',experimental='Fourparts positive, explicit experimental if strict fails.')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(v))
    else:save(path,v)
