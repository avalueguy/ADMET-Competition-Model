"""Remote-only fit-nested MAE experts and convex/global-property stackers."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'4')
import hashlib,json,time,platform
import joblib,numpy as np,torch
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import linprog
from scipy import sparse
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import v371_mae_experts as shared

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
TASKS=shared.TASKS
CONFIGS=(dict(name='convex_mae_stack',family='stack',gate=False),
         dict(name='property_gated_mae_stack',family='stack',gate=True))
WEIGHTS=(.2,.45,.75,1.)
EXPERTS=('svr_C30','ridge_03','catboost_mae','xgb_median')
SEED=3721106
GATES={'caco2_wang':2,'ld50_zhu':0,'lipophilicity_astrazeneca':1,'ppbr_az':1,'solubility_aqsoldb':1}
BUDGET={'caco2_wang':650,'ld50_zhu':900,'lipophilicity_astrazeneca':900,'ppbr_az':800,'solubility_aqsoldb':1100}
sha=shared.sha
save=shared.save
emit=shared.emit
target=shared.target
clip=shared.clip

def signature():
    return hashlib.sha256((sha(__file__)+sha(shared.__file__)+sha(ROOT/'data_manifest.json')+json.dumps([CONFIGS,WEIGHTS,EXPERTS,SEED,GATES,BUDGET],sort_keys=True)).encode()).hexdigest()

def setup():
    assert platform.system()=='Linux' and torch.cuda.is_available(),'Fit only on the authorized remote GPU host'
    shared.setup()

def load():
    shared.ROOT=ROOT
    return shared.load()

def ctrl(task,part):return shared.ctrl(task,part)
def design(d,task,part):return shared.design(d,task,part)

def subdesign(a,tr,hold):
    return dict(fit=a['fit'][tr],query=a['fit'][hold],x=a['x'][tr],qx=a['x'][hold],source=a['source'])

def kernel(d,a,task,part,tag):
    path=OUT/'kernels'/task/part/(tag+'.npz')
    if path.exists():
        z=np.load(path);assert str(z['signature'])==signature()
        for k in ('fit','query'):np.testing.assert_array_equal(z[k],a[k])
        return z['train'],z['query_kernel']
    k,kq,_=shared.kernels(d,a)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,signature=signature(),fit=a['fit'],query=a['query'],train=k,query_kernel=kq)
    return k,kq

def read_kernel(d,a,task,part,tag):
    path=OUT/'kernels'/task/part/(tag+'.npz')
    z=np.load(path);assert str(z['signature'])==signature()
    for key in ('fit','query'):np.testing.assert_array_equal(z[key],a[key])
    return z['train'],z['query_kernel']

def tree_inputs(d,a,cols=None):
    raw=d['fingerprints']
    if cols is None:
        var=raw[d['mi'][a['fit']]].astype(float).var(0)
        cols=np.sort(np.argsort(-var,kind='stable')[:768])
    def clean(x):return np.where(np.isfinite(x),np.clip(x,-1e18,1e18),np.nan).astype(np.float32)
    x=clean(np.c_[a['x'],raw[d['mi'][a['fit']]][:,cols]])
    qx=clean(np.c_[a['qx'],raw[d['mi'][a['query']]][:,cols]])
    # CatBoost uses explicit missing values supported by its GPU numeric features.
    keep=np.flatnonzero(np.nanstd(x.astype(float),axis=0)>1e-7)
    return x,qx,cols,keep

def fit_base(d,a,task,part,tag):
    path=OUT/'base'/task/part/tag/'state.joblib'
    if path.exists():
        s=joblib.load(path);assert s['signature']==signature()
        for k in ('fit','query'):np.testing.assert_array_equal(a[k],s[k])
        return s
    start=time.monotonic();fit,q=a['fit'],a['query']
    assert np.isfinite(d['y'][fit]).all() and not np.intersect1d(fit,q).size
    assert not np.intersect1d(d['rg'][fit],d['rg'][q]).size or part=='final' and tag=='full'
    y,c,s=target(d['y'][fit]);k,kq=kernel(d,a,task,part,tag)
    svr=SVR(kernel='precomputed',C=30.,epsilon=.03,tol=1e-4,max_iter=200000,cache_size=1000).fit(k,y)
    assert svr.fit_status_==0
    coef=cho_solve(cho_factor(k.astype(float)+np.eye(len(fit))*.3),y)
    x,qx,cols,keep=tree_inputs(d,a)
    cat=CatBoostRegressor(loss_function='MAE',iterations=BUDGET[task],depth=6,learning_rate=.035,l2_leaf_reg=12,
        random_seed=SEED,task_type='GPU',devices='0',thread_count=4,verbose=False,allow_writing_files=False,
        border_count=64,bootstrap_type='Bayesian',bagging_temperature=.5)
    cat.fit(x[:,keep],y)
    xgb=XGBRegressor(objective='reg:quantileerror',quantile_alpha=.5,n_estimators=BUDGET[task],max_depth=4,
        learning_rate=.035,min_child_weight=10,reg_lambda=25.,reg_alpha=.2,subsample=.85,colsample_bytree=.8,
        max_bin=128,tree_method='hist',device='cuda',n_jobs=4,random_state=SEED)
    xgb.fit(x,y,verbose=False)
    p=np.column_stack([svr.predict(kq),kq@coef,cat.predict(qx[:,keep]),xgb.predict(qx)])*s+c
    p=clip(task,p);assert np.isfinite(p).all()
    state=dict(signature=signature(),fit=fit,query=q,center=c,scale=s,svr=svr,coef=coef,cat=cat,xgb=xgb,columns=cols,keep=keep,p=p)
    path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(state,path,compress=3)
    save(path.with_name('metadata.json'),dict(signature=signature(),model_sha256=sha(path),seconds=time.monotonic()-start,fit=len(fit),query=len(q),models=4))
    emit('base_complete',task=task,part=part,tag=tag,seconds=time.monotonic()-start,models=4)
    return state

def replay_base(d,a,task,part,tag):
    path=OUT/'base'/task/part/tag/'state.joblib';s=joblib.load(path)
    assert s['signature']==signature()
    for key in ('fit','query'):np.testing.assert_array_equal(a[key],s[key])
    _,kq=read_kernel(d,a,task,part,tag)
    _,qx,cols,_=tree_inputs(d,a,s['columns']);np.testing.assert_array_equal(cols,s['columns'])
    p=np.column_stack([s['svr'].predict(kq),kq@s['coef'],s['cat'].predict(qx[:,s['keep']]),s['xgb'].predict(qx)])*s['scale']+s['center']
    return clip(task,p)

def fit_simplex(p,y,prior,penalty):
    """Convex MAE fit; weights nonnegative, sum1, small regularized intercept."""
    z,c,s=target(y);p=(p-c)/s;n,e=p.shape
    # Variables: expert weights, intercept, absolute sample errors, abs weight offsets, abs intercept.
    total=e+1+n+e+1;bi=e;ui=e+1;vi=ui+n;vb=vi+e
    cost=np.zeros(total);cost[ui:vi]=1/n;cost[vi:vb]=penalty;cost[vb]=.03
    A=sparse.lil_matrix((2*n+2*e+2,total));rhs=np.zeros(A.shape[0])
    A[:n,:e]=p;A[:n,bi]=1;A[:n,ui:vi]=-sparse.eye(n);rhs[:n]=z
    A[n:2*n,:e]=-p;A[n:2*n,bi]=-1;A[n:2*n,ui:vi]=-sparse.eye(n);rhs[n:2*n]=-z
    for j in range(e):
        A[2*n+j,j]=1;A[2*n+j,vi+j]=-1;rhs[2*n+j]=prior[j]
        A[2*n+e+j,j]=-1;A[2*n+e+j,vi+j]=-1;rhs[2*n+e+j]=-prior[j]
    A[-2,bi]=1;A[-2,vb]=-1;A[-1,bi]=-1;A[-1,vb]=-1
    equal=sparse.lil_matrix((1,total));equal[0,:e]=1
    result=linprog(cost,A_ub=A.tocsr(),b_ub=rhs,A_eq=equal.tocsr(),b_eq=[1.],
        bounds=[(0,1)]*e+[(-.1,.1)]+[(0,None)]*(n+e+1),method='highs')
    assert result.success,result.message
    w=result.x[:e];b=float(result.x[bi]*s)
    assert abs(w.sum()-1)<1e-7 and w.min()>-1e-7
    return dict(weights=w,bias=b,objective=float(result.fun),success=True,n=n,penalty=penalty)

def gate_values(d,rows,task,median=None):
    values=d['physical'][d['mi'][rows],GATES[task]].astype(float)
    if median is None:median=float(np.nanmedian(values))
    return np.where(np.isfinite(values),values,median),median

def fit_stack(d,a,task,oof,full,c):
    y=d['y'][a['fit']];global_s=fit_simplex(oof,y,np.ones(4)/4,.01)
    p=full@global_s['weights']+global_s['bias'];state=dict(global_stack=global_s,locals=[],edges=[],gate_median=None)
    if c['gate']:
        v,median=gate_values(d,a['fit'],task);qv,_=gate_values(d,a['query'],task,median);edges=np.unique(np.quantile(v,[1/3,2/3]))
        b=np.searchsorted(edges,v,side='right');qb=np.searchsorted(edges,qv,side='right');state.update(edges=edges,gate_median=median)
        for index in range(len(edges)+1):
            ix=np.flatnonzero(b==index);loc=global_s
            if len(ix)>=60:loc=fit_simplex(oof[ix],y[ix],global_s['weights'],.03)
            # Fixed half shrinkage keeps small property bins conservative.
            w=.5*global_s['weights']+.5*loc['weights'];off=.5*global_s['bias']+.5*loc['bias']
            p[qb==index]=full[qb==index]@w+off
            state['locals'].append(dict(bin=index,weights=w,bias=off,fit_count=len(ix),raw=loc))
    return clip(task,p),state

def predict(d,task,part,c):
    setup();folder=OUT/'tasks'/task/part/c['name'];path=folder/'predictions.npz';sig=signature()
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig;return z['p']
    a=design(d,task,part);fit=a['fit'];oof=np.full((len(fit),4),np.nan);splits=[]
    for i,(tr,hold) in enumerate(GroupKFold(3).split(fit,groups=d['rg'][fit])):
        sub=subdesign(a,tr,hold);s=fit_base(d,sub,task,part,'inner'+str(i));oof[hold]=s['p']
        splits.append(dict(train=tr,held=hold))
    assert np.isfinite(oof).all()
    full=fit_base(d,a,task,part,'full')['p'];p,state=fit_stack(d,a,task,oof,full,c)
    state.update(signature=sig,config=c,fit=fit,query=a['query'],splits=splits,oof=oof,full=full)
    folder.mkdir(parents=True,exist_ok=True);joblib.dump(state,folder/'model.joblib',compress=3)
    np.savez_compressed(path,signature=sig,fit=fit,query=a['query'],p=p)
    save(folder/'metadata.json',dict(signature=sig,task=task,part=part,config=c,model_sha256=sha(folder/'model.joblib'),predictions_sha256=sha(path),target_context_excluded=True,inner_fit_preprocessing=True))
    emit('stack_complete',task=task,part=part,config=c['name'],weights=state['global_stack']['weights'].tolist())
    return p

def replay(d,task,part,c):
    a=design(d,task,part);s=joblib.load(OUT/'tasks'/task/part/c['name']/'model.joblib');assert s['signature']==signature()
    full=replay_base(d,a,task,part,'full');g=s['global_stack'];p=full@g['weights']+g['bias']
    if c['gate']:
        qv,_=gate_values(d,a['query'],task,s['gate_median']);qb=np.searchsorted(s['edges'],qv,side='right')
        for r in s['locals']:
            ix=qb==r['bin'];p[ix]=full[ix]@r['weights']+r['bias']
    return clip(task,p)

def freeze():
    value=dict(signature=signature(),configs=CONFIGS,weights=WEIGHTS,experts=EXPERTS,seed=SEED,budget=BUDGET,gate_physical_index=GATES,
        baseline='v364:51.269',all_fitting_remote=True,external_data=False,test_labels_used=False,
        selection='Max-min gain fold0/1; unique candidate frozen before fold2/valid; no fallback.',
        fit='Three inner scaffold folds, with feature scaling/selection rebuilt on each inner fit. Convex original-scale MAE weights from inner OOF only; no historical predictions as fitting inputs.',
        gated='Predetermined task physical axis, outerfit quantile thirds, OOF-derived local simplex weights, fixed50percent shrink to global.',
        architecture='Median SVR, kernel ridge, GPU CatBoost MAE, GPU XGBoost median. Fixed task budgets, no query early stopping.',
        context='Other official task measurements permitted; current target and its mask excluded; other valid observations only for final.',
        limitations='Public labels historically reused; baseline historical OOF partly proxy; outer group folds have concentrated scaffolds. Confirmation is not a fresh untouched holdout.',
        strict='All4positive, developmental minimum>1.2percent, four-part mean>1.2percent, both500group bootstrap90lower>0.',
        experimental='All4positive even if strict uncertainty fails, explicitly experimental; five10percent goal separately reported.')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(value))
    else:save(path,value)
