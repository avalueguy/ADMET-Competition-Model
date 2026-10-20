"""Remote sparse substructure kernels without folding to a fixed fingerprint size.

Vocabulary and descriptor statistics use fitting rows only. Query norms include
unseen substructures, so unseen chemistry is not spuriously made more similar.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import os

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.svm import SVC, SVR
from rdkit import Chem, rdBase
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetMorganFeatureAtomInvGen

from admet_v8_models import load_task as load8
from admet_v6_models import Preprocess, Target


def sparse_fingerprints(smiles):
    generators=[GetMorganGenerator(radius=r,includeChirality=True) for r in [1,2,3]]
    generators.append(GetMorganGenerator(radius=2,includeChirality=True,atomInvariantsGenerator=GetMorganFeatureAtomInvGen()))
    result=[[] for _ in generators]
    for smi in smiles:
        mol=Chem.MolFromSmiles(smi)
        for j,g in enumerate(generators):
            result[j].append({int(k):int(v) for k,v in g.GetSparseCountFingerprint(mol).GetNonzeroElements().items()} if mol else {})
    return result


def load_task(task):
    d=load8(task)
    schema=dict(version=11,rdkit=rdBase.rdkitVersion,representations=['ECFP1','ECFP2','ECFP3','FCFP2'],includeChirality=True)
    key=hashlib.sha256(json.dumps([schema,d['smiles']],sort_keys=True).encode()).hexdigest()
    path=Path('admet_model_outputs_v11/features')/(task+'.json')
    if path.exists():
        obj=json.loads(path.read_text());assert obj['key']==key
        fps=[[{int(k):int(v) for k,v in row.items()} for row in view] for view in obj['fingerprints']]
    else:
        fps=sparse_fingerprints(d['smiles']);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.json')
        tmp.write_text(json.dumps(dict(key=key,fingerprints=fps),separators=(',',':')));os.replace(tmp,path)
    d.update(sparse_views=fps,v11_feature_key=key)
    return d


def fit_vocabulary(rows):
    return {key:i for i,key in enumerate(sorted({k for row in rows for k in row}))}


def encode(rows,vocabulary,binary):
    rr=[];cc=[];vv=[];norm=[]
    for i,row in enumerate(rows):
        norm.append(float(len(row) if binary else sum(float(v)**2 for v in row.values())))
        for key,value in row.items():
            col=vocabulary.get(key)
            if col is not None:rr.append(i);cc.append(col);vv.append(1. if binary else float(value))
    matrix=sparse.csr_matrix((np.array(vv,dtype=np.float32),(rr,cc)),shape=(len(rows),len(vocabulary)),dtype=np.float32)
    return matrix,np.asarray(norm,dtype=np.float32)


def references(d,ix):
    result=[]
    # Fixed view weighting; not fitted from validation or test chemistry.
    for view,binary,weight in [(0,True,.10),(1,True,.25),(2,True,.20),(1,False,.15),(3,True,.20)]:
        rows=[d['sparse_views'][view][int(i)] for i in ix];vocab=fit_vocabulary(rows);matrix,norm=encode(rows,vocab,binary)
        result.append(dict(view=view,binary=binary,weight=weight,vocabulary=vocab,matrix=matrix,norm=norm))
    return result


def kernel(d,ix,bundle):
    pre=bundle['pre'];desc=pre.transform(d['desc'][ix]);ref=bundle['reference_desc']
    out=np.zeros((len(ix),len(ref)),dtype=np.float64)
    for r in bundle['references']:
        x,norm=encode([d['sparse_views'][r['view']][int(i)] for i in ix],r['vocabulary'],r['binary'])
        for start in range(0,len(ix),256):
            dot=(x[start:start+256]@r['matrix'].T).toarray()
            den=norm[start:start+256,None]+r['norm'][None,:]-dot
            out[start:start+256]+=r['weight']*dot/np.maximum(den,1e-8)
    for start in range(0,len(ix),256):
        a=desc[start:start+256]
        dist=np.maximum((a*a).sum(1)[:,None]+(ref*ref).sum(1)[None,:]-2*a@ref.T,0)/a.shape[1]
        out[start:start+256]+=.05*np.exp(-dist)+.05*np.exp(-dist/4)
    return out


def sample_weights(groups):
    freq=Counter(groups);w=np.array([freq[g]**(-.25) for g in groups],dtype=float)
    # Solve for a mean-one scale without moving outside the stated bounds.
    lo,hi=0.,2./w.min()
    for _ in range(64):
        mid=(lo+hi)/2
        if np.clip(mid*w,.5,2.).mean()<1:lo=mid
        else:hi=mid
    return np.clip((lo+hi)/2*w,.5,2.)


def fit(family,d,ix,y,metric,protocol):
    from admet_v6_runner import remote_only
    remote_only();assert family in protocol['families']
    pre=Preprocess().fit(d['desc'][ix]);cls=metric in ['AUROC','AUPRC']
    tf=None if cls else Target(np.asarray(y),metric)
    yy=np.asarray(y) if cls else tf.encode(y)
    bundle=dict(family=family,metric=metric,classification=cls,target=tf,feature_key=d['v11_feature_key'],
        pre=pre,reference_desc=pre.transform(d['desc'][ix]),references=references(d,ix),models=[],reference=np.asarray(ix))
    k=kernel(d,ix,bundle);np.testing.assert_allclose(k,k.T,atol=2e-7)
    weights=sample_weights(d['groups'][ix]) if family=='sparse_scaffold' else np.ones(len(ix))
    for c in protocol['kernel_cs']:
        model=SVC(kernel='precomputed',C=c,class_weight='balanced',cache_size=1536) if cls else SVR(kernel='precomputed',C=c,epsilon=.03,cache_size=1536)
        model.fit(k,yy,sample_weight=weights);bundle['models'].append(model)
    bundle['weight_range']=[float(weights.min()),float(weights.max())]
    bundle['train_rows']=len(ix)
    return bundle


def predict(bundle,d,ix):
    assert bundle['feature_key']==d['v11_feature_key']
    k=kernel(d,ix,bundle);values=[]
    for model in bundle['models']:
        values.append(model.decision_function(k) if bundle['classification'] else model.predict(k))
    p=np.mean(values,axis=0)
    return expit(p) if bundle['classification'] else bundle['target'].decode(p)
