"""Official-data endpoint experts; frozen v72 remains the submission anchor.

Test labels are never parsed. New experts are evaluated on the existing global
scaffold folds and official valid. Historical ensemble arrays are explicitly
development proxies, not independent OOF or an exact v72 control for all tasks.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator, GetAtomPairGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
import admet_v67_data as b

ROOT=b.ROOT
OUT=ROOT/'admet_model_outputs_v73'
FAMILIES=('physchem', 'multiview')
PARTS=('fold0','fold1','fold2','valid')
SEED=732026

def signature():
    return hashlib.sha256((b.signature()+b.sha(__file__)).encode()).hexdigest()

def features(d):
    folder=OUT/'features';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'views.npz';schema=dict(signature=signature(),rdkit=rdBase.rdkitVersion,
        smiles_sha256=hashlib.sha256(json.dumps(d['smiles']).encode()).hexdigest())
    if path.exists():
        assert json.loads((folder/'schema.json').read_text())==schema
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    # Whole input and largest-parent views retain complementary salt information.
    gens=[GetMorganGenerator(radius=3,fpSize=2048,includeChirality=True),
          GetRDKitFPGenerator(fpSize=2048,maxPath=7),
          GetAtomPairGenerator(fpSize=2048,includeChirality=True)]
    fns=[(n,f) for n,f in Descriptors.descList if n not in {'Ipc','AvgIpc','BalabanJ'}]
    n=len(d['smiles']); extra=np.zeros((n,8192),np.float32)
    parent=np.full((n,len(fns)+5),np.nan,np.float32);priors=np.zeros((n,2),np.float32)
    exceptions=[]
    for i,sm in enumerate(d['smiles']):
        mol,_=b.parse_molecule(sm)
        for j,gen in enumerate(gens):extra[i,j*2048:(j+1)*2048]=np.log1p(gen.GetCountFingerprintAsNumPy(mol))
        fragments=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=False)
        par=max(fragments,key=lambda m:(sum(a.GetAtomicNum()==6 for a in m.GetAtoms()),m.GetNumHeavyAtoms()))
        par.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(par,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,catchErrors=True)
        extra[i,6144:]=np.log1p(gens[0].GetCountFingerprintAsNumPy(par))
        for j,(_,fn) in enumerate(fns):
            try:
                value=float(fn(par))
                if np.isfinite(value) and abs(value)<1e10:parent[i,j]=value
            except Exception:pass
        lp=Crippen.MolLogP(par);mw=Descriptors.MolWt(par);rb=Lipinski.NumRotatableBonds(par)
        ap=sum(a.GetIsAromatic() for a in par.GetAtoms())/max(1,par.GetNumHeavyAtoms())
        # Fixed analytical expression from intrinsic structure, no external labels.
        esol=.16-.63*lp-.0062*mw+.066*rb-.74*ap
        priors[i]=[lp,esol]
        parent[i,-5:]=[len(fragments),Chem.GetFormalCharge(mol),Chem.GetFormalCharge(par),mol.GetNumHeavyAtoms()-par.GetNumHeavyAtoms(),esol]
        if i%10000==0:b.emit('v73_features',done=i,total=n)
    result=dict(extra=extra,parent=parent,priors=priors)
    np.savez_compressed(path,**result);b.save(folder/'schema.json',schema)
    b.save(folder/'audit.json',dict(official_structures_only=True,fit_free_features=True,
        whole_input_preserved=True,parent_is_additional_view=True,extra_fingerprint_dimensions=8192))
    return result

def rows(d,k,part):
    folds=np.load(b.OUT/'folds.npy');task=d['tid']==k
    if part=='valid':fit=np.flatnonzero(task&(d['split']==0));q=np.flatnonzero(task&(d['split']==1))
    elif part=='final':fit=np.flatnonzero(task&(d['split']<2));q=np.flatnonzero(task&(d['split']==2))
    else:
        f=int(part[-1]);fit=np.flatnonzero(task&(d['split']==0)&(folds!=f));q=np.flatnonzero(task&(d['split']==0)&(folds==f))
    b.assert_excluded(d,fit,q)
    return fit,q

def matrix(d,v,ix,family):
    mol=d['mi'][ix]
    desc=np.concatenate([d['x'][mol,4096:],v['parent'][mol],v['priors'][mol]],1)
    if family=='physchem':return np.nan_to_num(desc,nan=np.nan,posinf=np.nan,neginf=np.nan).astype(np.float32)
    return np.concatenate([d['x'][mol,:4096],v['extra'][mol],desc],1).astype(np.float32)

def prior(d,v,k,ix,family):
    if family!='physchem':return np.zeros(len(ix))
    task=d['tasks'][k]
    col=0 if task=='lipophilicity_astrazeneca' else 1 if task=='solubility_aqsoldb' else None
    return np.zeros(len(ix)) if col is None else v['priors'][d['mi'][ix],col].astype(float)

def config(metric,family,iterations,seed):
    cls=metric in ('AUROC','AUPRC')
    return dict(iterations=iterations,depth=5 if family=='physchem' else 6,
        learning_rate=.025 if family=='physchem' else .035,l2_leaf_reg=12 if family=='physchem' else 20,
        loss_function='Logloss' if cls else 'RMSE',
        eval_metric='AUC' if metric=='AUROC' else 'PRAUC' if metric=='AUPRC' else 'MAE' if metric=='MAE' else 'RMSE',
        task_type='GPU',devices='0',random_seed=seed,thread_count=4,verbose=False,
        allow_writing_files=False,border_count=128,bootstrap_type='Bernoulli',subsample=.85,
        nan_mode='Min',boosting_type='Plain')

def fit_predict(d,v,k,part,family,iterations=2200,seed=SEED):
    fit,q=rows(d,k,part);task=d['tasks'][k];metric=d['metrics'][k]
    folder=OUT/task/part/family/str(seed);folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'predictions.npz';sig=signature()
    if cache.exists():
        z=np.load(cache);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q)
        meta=json.loads((folder/'metadata.json').read_text());assert meta['model_sha256']==b.sha(folder/'model.cbm')
        return z['p'].copy()
    start=time.monotonic();cls=metric in ('AUROC','AUPRC')
    x=matrix(d,v,fit,family);qx=matrix(d,v,q,family)
    pr=prior(d,v,k,fit,family);qp=prior(d,v,k,q,family)
    tf=b.target(d['y'][fit]-pr if metric=='MAE' else d['y'][fit],metric)
    y=b.encode(d['y'][fit]-pr if metric=='MAE' else d['y'][fit],tf)
    cfg=config(metric,family,iterations,seed);net=(CatBoostClassifier if cls else CatBoostRegressor)(**cfg)
    _,inv,cnt=np.unique(d['canonical'][d['mi'][fit]],return_inverse=True,return_counts=True)
    sw=1/cnt[inv];sw/=sw.mean()
    if part!='final':
        vy=b.encode(d['y'][q]-qp if metric=='MAE' else d['y'][q],tf)
        net.fit(x,y,sample_weight=sw,eval_set=(qx,vy),use_best_model=True,early_stopping_rounds=220)
    else:net.fit(x,y,sample_weight=sw)
    p=net.predict_proba(qx)[:,1] if cls else net.predict(qx)
    if metric=='MAE':p=p*tf['scale']+tf['center']+qp
    assert np.isfinite(p).all();net.save_model(str(folder/'model.cbm'))
    np.savez_compressed(cache,p=p,fit=fit,query=q,signature=sig)
    serial=lambda t:{key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in t.items()}
    b.save(folder/'metadata.json',dict(signature=sig,task=task,part=part,family=family,seed=seed,
        config=cfg,trees=net.tree_count_,target=serial(tf),fit_rows=len(fit),query_rows=len(q),
        model_sha256=b.sha(folder/'model.cbm'),predictions_sha256=b.sha(cache),
        test_labels_used=False,pretrained_weights=False,external_data=False,independent_reload_pending=True))
    b.emit('v73_fit',task=task,part=part,family=family,seed=seed,trees=net.tree_count_,
        score=b.metric_score(metric,d['y'][q],p) if part!='final' else None,seconds=round(time.monotonic()-start))
    return p

def develop(d,v,tasks,families,parts):
    OUT.mkdir(exist_ok=True)
    b.save(OUT/'protocol.json',dict(signature=signature(),anchor='v72',anchor_official_score_user_reported=50.628,
        tasks=tasks,families=list(families),parts=list(parts),max_trees=2200,seed=SEED,
        selection_uses_development_labels=True,independent_unbiased_evaluation=False,
        training_data='official train and valid only',test_labels_used=False,external_data=False,pretrained_weights=False))
    for task in tasks:
        k=d['tasks'].index(task)
        for family in families:
            for part in parts:fit_predict(d,v,k,part,family)
    b.save(OUT/'development_completed.json',dict(status='COMPLETE',signature=signature(),tasks=tasks,families=list(families),parts=list(parts)))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','develop']);p.add_argument('--tasks',nargs='+')
    p.add_argument('--families',nargs='+',default=FAMILIES,choices=FAMILIES);p.add_argument('--parts',nargs='+',default=PARTS,choices=PARTS)
    a=p.parse_args();assert os.name!='nt' and __import__('torch').cuda.is_available()
    d=b.build();v=features(d)
    if a.stage=='develop':develop(d,v,a.tasks or d['tasks'],a.families,a.parts)

if __name__=='__main__':main()
