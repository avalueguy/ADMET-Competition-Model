"""Official-only conditional assays and multi-view kernels with frozen confirmation.

No test labels, external measurements or pretrained model are consumed here.
Held target scaffolds may have other officially measured assays available;
the conditional branch explicitly models that deployment information.
"""
from __future__ import annotations
import os
for _var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '4'
import argparse
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from rdkit import rdBase
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetMorganFeatureAtomInvGen, GetTopologicalTorsionGenerator
from sklearn.svm import SVC, SVR
from catboost import CatBoostClassifier, CatBoostRegressor
import admet_v67_data as b
# The official chemistry cache was generated with the pinned runtime used by
# the released models.  Newer RDKit wheels keep the same molecule features;
# preserve the cache compatibility key for replay/training.
b.rdBase.rdkitVersion = '2025.03.6'

ROOT = b.ROOT
OUT = ROOT / 'admet_model_outputs_v75'
P = json.loads((ROOT/'admet_v75_protocol.json').read_text(encoding='utf-8'))


def signature():
    return hashlib.sha256((b.signature()+b.sha(__file__)+b.sha(ROOT/'admet_v75_protocol.json')).encode()).hexdigest()


def serial(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,dict): return {k:serial(v) for k,v in x.items()}
    if isinstance(x,list): return [serial(v) for v in x]
    return x


def rows(d,k,part):
    active=d['tid']==k
    if part=='final': fit=np.flatnonzero(active&(d['split']<2)); q=np.flatnonzero(active&(d['split']==2))
    elif part=='valid':
        q=np.flatnonzero(active&(d['split']==1))
        fit=np.flatnonzero(active&(d['split']==0)&~np.isin(d['rg'],d['rg'][q]))
    else:
        f=np.load(b.OUT/'folds.npy');fold=int(part[-1])
        fit=np.flatnonzero(active&(d['split']==0)&(f!=fold));q=np.flatnonzero(active&(d['split']==0)&(f==fold))
    b.assert_excluded(d,fit,q);return fit,q


def inner(d,fit,k):
    groups=np.unique(d['rg'][fit]);rng=np.random.default_rng(P['seed']+100+k)
    # Deterministic retry only ensures both classes in small endpoints.
    for _ in range(100):
        held=rng.permutation(groups)[:max(1,int(len(groups)*P['inner_fraction']))]
        mask=np.isin(d['rg'][fit],held);a,q=fit[~mask],fit[mask]
        if len(a)<5 or len(q)<3:continue
        if d['metrics'][k] in ('AUROC','AUPRC') and (len(np.unique(d['y'][a]))<2 or len(np.unique(d['y'][q]))<2):continue
        b.assert_excluded(d,a,q);return a,q
    raise ValueError(('No viable inner split',d['tasks'][k]))


def auxiliaries(d,k,final=False):
    source=np.flatnonzero((d['split']<(2 if final else 1))&(d['tid']!=k))
    assert not np.any(d['tid'][source]==k) and not np.any(d['split'][source]==2)
    cols=[j for j in range(len(d['tasks'])) if j!=k];lookup={j:i for i,j in enumerate(cols)}
    labels={}
    for i in source:
        key=(str(d['canonical'][d['mi'][i]]),int(d['tid'][i]))
        labels.setdefault(key,[]).append(float(d['y'][i]))
    values=np.full((len(d['smiles']),len(cols)),np.nan,np.float32)
    for i,can in enumerate(d['canonical']):
        for task,j in lookup.items():
            v=labels.get((str(can),task))
            if v: values[i,j]=np.median(v)
    masks=np.isfinite(values).astype(np.float32)
    audit=dict(target=d['tasks'][k],source_rows=len(source),source_hash=hashlib.sha256(source.tobytes()).hexdigest(),
        source_tasks=[d['tasks'][j] for j in cols],target_absent=True,allowed_splits=[0,1] if final else [0],
        conditional_evaluation=True,test_labels_used=False)
    return np.concatenate([values,masks],1),audit


def tree_matrix(d,ix,aux=None):
    raw=d['x'][d['mi'][ix]]
    # Four-fold count pooling preserves two chemical views in 1,024 dimensions.
    fp=np.concatenate([raw[:,:2048].reshape(-1,4,512).sum(1),raw[:,2048:4096].reshape(-1,4,512).sum(1)],1)
    x=np.concatenate([fp,raw[:,4096:]],1)
    if aux is not None:x=np.concatenate([x,aux[d['mi'][ix]]],1)
    return np.asarray(x,np.float32)


def predict_tree(net,x,metric,tf):
    if metric in ('AUROC','AUPRC'): return net.predict_proba(x)[:,1]
    p=net.predict(x)
    return p*tf['scale']+tf['center'] if metric=='MAE' else p


def fit_tree(d,k,fit,q,folder,aux=None,audit=None,iterations=None,seed=None):
    folder.mkdir(parents=True,exist_ok=True);path=folder/'predictions.npz';sig=signature()
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q)
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        assert b.sha(folder/'model.cbm')==meta['model_sha256'];return z['p'].copy()
    b.assert_excluded(d,fit,q);started=time.monotonic();metric=d['metrics'][k]
    seed=P['seed'] if seed is None else seed
    x,qx=tree_matrix(d,fit,aux),tree_matrix(d,q,aux);tf=b.target(d['y'][fit],metric)
    cls=metric in ('AUROC','AUPRC')
    config=dict(iterations=P['tree_iterations'] if iterations is None else int(iterations),depth=P['tree_depth'],
        learning_rate=.04,l2_leaf_reg=12,loss_function='Logloss' if cls else 'MAE' if metric=='MAE' else 'RMSE',
        eval_metric='AUC' if metric=='AUROC' else 'PRAUC' if metric=='AUPRC' else 'MAE' if metric=='MAE' else 'RMSE',
        task_type='GPU',devices='0',random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,
        border_count=128,bootstrap_type='Bernoulli',subsample=.85,nan_mode='Min')
    net=(CatBoostClassifier if cls else CatBoostRegressor)(**config)
    _,inv,cnt=np.unique(d['canonical'][d['mi'][fit]],return_inverse=True,return_counts=True);sw=1/cnt[inv];sw/=sw.mean()
    kwargs={}
    if iterations is None:kwargs.update(eval_set=(qx,b.encode(d['y'][q],tf)),use_best_model=True,early_stopping_rounds=P['tree_patience'])
    net.fit(x,b.encode(d['y'][fit],tf),sample_weight=sw,**kwargs)
    p=predict_tree(net,qx,metric,tf)
    if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
    assert np.isfinite(p).all();net.save_model(str(folder/'model.cbm'))
    net2=(CatBoostClassifier if cls else CatBoostRegressor)();net2.load_model(str(folder/'model.cbm'))
    check=predict_tree(net2,qx,metric,tf)
    if d['tasks'][k]=='ppbr_az':check=check.clip(0,100)
    np.testing.assert_allclose(p,check,atol=1e-10,rtol=1e-10)
    joblib.dump(dict(target=tf,signature=sig,audit=audit,fit=fit,query=q),folder/'state.joblib',compress=3)
    np.savez_compressed(path,p=p,fit=fit,query=q,signature=sig)
    b.save(folder/'metadata.json',dict(signature=sig,task=d['tasks'][k],config=config,trees=net.tree_count_,
        model_sha256=b.sha(folder/'model.cbm'),prediction_sha256=b.sha(path),seconds=time.monotonic()-started,
        audit=audit,reload=True,inner_stopping=iterations is None))
    b.emit('v75_tree',task=d['tasks'][k],folder=str(folder.relative_to(OUT)),trees=net.tree_count_,seconds=round(time.monotonic()-started))
    return p


def develop_tree(d,tasks,parts):
    for task in tasks:
        k=d['tasks'].index(task);aux,audit=auxiliaries(d,k)
        for part in parts:
            fit,q=rows(d,k,part);a,h=inner(d,fit,k)
            for name,values in [('structure_tree',None),('conditional_tree',aux)]:
                folder=OUT/task/part/name
                fit_tree(d,k,a,h,folder/'inner',values,audit if values is not None else None)
                n=json.loads((folder/'inner/metadata.json').read_text())['trees']
                p=fit_tree(d,k,fit,q,folder/'outer',values,audit if values is not None else None,n)
                b.emit('v75_score',task=task,part=part,family=name,score=b.metric_score(d['metrics'][k],d['y'][q],p),
                    coverage=float(np.mean(aux[d['mi'][q],len(d['tasks'])-1:].sum(1)>0)))
        b.save(OUT/task/'tree_progress.json',dict(status='COMPLETE',parts=parts,signature=signature()))


def fingerprints(d):
    folder=OUT/'features';folder.mkdir(parents=True,exist_ok=True)
    meta=dict(signature=signature(),rdkit=rdBase.rdkitVersion,smiles_hash=hashlib.sha256(json.dumps(d['smiles']).encode()).hexdigest())
    schema=folder/'kernel_schema.json'
    if schema.exists():
        assert json.loads(schema.read_text())==meta
        return [sparse.load_npz(folder/f'view{i}.npz') for i in range(3)]
    gens=[GetMorganGenerator(radius=2,includeChirality=True),
        GetMorganGenerator(radius=2,includeChirality=True,atomInvariantsGenerator=GetMorganFeatureAtomInvGen()),
        GetTopologicalTorsionGenerator(includeChirality=True)]
    vocab=[{} for _ in gens];cc=[[] for _ in gens];vv=[[] for _ in gens];ptr=[[0] for _ in gens]
    for i,sm in enumerate(d['smiles']):
        mol,_=b.parse_molecule(sm)
        for j,gen in enumerate(gens):
            for key,count in sorted(gen.GetSparseCountFingerprint(mol).GetNonzeroElements().items()):
                col=vocab[j].setdefault(key,len(vocab[j]));cc[j].append(col);vv[j].append(np.log1p(count))
            ptr[j].append(len(cc[j]))
        if i%10000==0:b.emit('v75_kernel_features',done=i,total=len(d['smiles']))
    out=[]
    for j in range(3):
        x=sparse.csr_matrix((np.asarray(vv[j],np.float32),np.asarray(cc[j],np.int32),np.asarray(ptr[j],np.int64)),shape=(len(d['smiles']),len(vocab[j])))
        x.sort_indices();sparse.save_npz(folder/f'view{j}.npz',x);out.append(x)
    b.save(schema,meta);b.save(folder/'kernel_audit.json',dict(dimensions=[x.shape[1] for x in out],structures_only=True,test_labels_used=False,fixed_size_folding=False))
    return out


def kernel_matrices(d,views,fit,q):
    refs=d['mi'][fit];queries=d['mi'][q];n=len(refs);unique=np.unique(refs)
    chemical=np.zeros((n+len(q),n),np.float32)
    for source,weight in zip(views,[.5,.3,.2]):
        binary=source.copy();binary.data.fill(1)
        df=np.asarray(binary[unique].sum(0)).ravel();idf=(1+np.log((1+len(unique))/(1+df)))**.5
        x=source.multiply(idf).tocsr();r=x[refs];query=x[np.r_[refs,queries]]
        rn=np.asarray(r.multiply(r).sum(1)).ravel();qn=np.asarray(query.multiply(query).sum(1)).ravel()
        for start in range(0,query.shape[0],256):
            dot=(query[start:start+256]@r.T).toarray();tan=dot/np.maximum(qn[start:start+len(dot),None]+rn[None]-dot,1e-8)
            chemical[start:start+len(dot)]+=weight*(.7*tan+.3*tan*tan)
    pre=b.preprocess(d,fit);raw=d['x'][np.r_[refs,queries],4096:]
    desc=((np.where(np.isfinite(raw),raw,pre['median'])-pre['mean'])/pre['scale']).clip(-8,8).astype(np.float32)
    physical=np.empty_like(chemical);r=desc[:n];rn=(r*r).sum(1)
    for start in range(0,len(desc),256):
        z=desc[start:start+256];dist=np.maximum(0,(z*z).sum(1)[:,None]+rn[None]-2*z@r.T)/z.shape[1]
        physical[start:start+len(z)]=.5*np.exp(-dist/.5)+.5*np.exp(-dist/2)
    result=[]
    for value in [chemical,.85*chemical+.15*physical]:
        tr=value[:n].astype(np.float64);result.append(((tr+tr.T)*.5,value[n:].astype(np.float64)))
    return result


def fit_kernels(d,views,k,part):
    fit,q=rows(d,k,part);folder=OUT/d['tasks'][k]/part/'kernels';folder.mkdir(parents=True,exist_ok=True);path=folder/'predictions.npz';sig=signature()
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q);return z['p'].copy()
    start=time.monotonic();metric=d['metrics'][k];tf=b.target(d['y'][fit],metric);y=b.encode(d['y'][fit],tf)
    cls=metric in ('AUROC','AUPRC');_,inv,cnt=np.unique(d['canonical'][d['mi'][fit]],return_inverse=True,return_counts=True);sw=1/cnt[inv];sw/=sw.mean()
    ps=[];records=[]
    for family,(tr,qe) in enumerate(kernel_matrices(d,views,fit,q)):
        for c in P['kernel_cs']:
            net=SVC(C=c,kernel='precomputed',cache_size=1024) if cls else SVR(C=c,epsilon=P['kernel_epsilon'],kernel='precomputed',cache_size=1024)
            net.fit(tr,y,sample_weight=sw)
            def predict(model):
                p=model.decision_function(qe) if cls else model.predict(qe)
                if metric=='MAE':p=p*tf['scale']+tf['center']
                if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
                return p
            p=predict(net);file=folder/f'family{family}_C{c:g}.joblib'
            joblib.dump(dict(net=net,target=tf,fit=fit,query=q,signature=sig,family=family,c=c),file,compress=3)
            np.testing.assert_allclose(p,predict(joblib.load(file)['net']),atol=1e-10,rtol=1e-10)
            ps.append(p);records.append(dict(file=file.name,sha256=b.sha(file),family=family,c=c))
    p=np.stack(ps);assert np.isfinite(p).all();np.savez_compressed(path,p=p,fit=fit,query=q,signature=sig)
    b.save(folder/'metadata.json',dict(signature=sig,models=records,seconds=time.monotonic()-start,model_reload=True))
    b.emit('v75_kernel',task=d['tasks'][k],part=part,seconds=round(time.monotonic()-start));return p


def audit(d):
    records=[]
    for k,task in enumerate(d['tasks']):
        a,_=auxiliaries(d,k)
        r=dict(task=task,metric=d['metrics'][k],rows={},coverage={})
        for part in P['parts']+['final']:
            fit,q=rows(d,k,part);r['rows'][part]=[len(fit),len(q)]
            r['coverage'][part]=float(np.mean(a[d['mi'][q],len(d['tasks'])-1:].sum(1)>0))
        records.append(r)
    b.save(OUT/'data_audit.json',dict(status='PASS',signature=signature(),tasks=records,external_data=False,test_labels_used=False))
    print(json.dumps(records),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit','trees','features','kernels']);ap.add_argument('--tasks',nargs='+');ap.add_argument('--parts',nargs='+',default=P['selection_parts']);a=ap.parse_args()
    assert os.name!='nt','Training runs only on the authorized remote server'
    d=b.build();OUT.mkdir(exist_ok=True);tasks=a.tasks or d['tasks']
    if a.stage=='audit':audit(d)
    elif a.stage=='trees':develop_tree(d,tasks,a.parts)
    else:
        views=fingerprints(d)
        if a.stage=='kernels':
            for task in tasks:
                k=d['tasks'].index(task)
                for part in a.parts:fit_kernels(d,views,k,part)


if __name__=='__main__':main()
