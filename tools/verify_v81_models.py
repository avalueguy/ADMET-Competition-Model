"""Independent replay and byte-level verification for v81."""
from pathlib import Path
import sys, json, csv, io, hashlib
from collections import Counter
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from scipy.optimize import minimize
from catboost import CatBoostClassifier

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import admet_v77_ranking as b
from verify_submission import verify
from repack_upload_template import field_spans

TASKS=('bioavailability_ma','cyp2d6_veith','pgp_broccatelli')
SEEDS=(802026,802027,802028)
OUT=ROOT/'admet_model_outputs_v81_final'

def ranklog(x):return logit(np.clip((rankdata(np.asarray(x))-.5)/len(x),.002,.998))
def calibrate(y,z):
    x=np.c_[np.ones(len(z)),z]
    def fun(t):
        eta=x@t;r=expit(eta)-y
        return (np.logaddexp(0,eta)-y*eta).mean()+.001*t[1]**2,x.T@r/len(r)+[0,.002*t[1]]
    return minimize(fun,[0.,1.],jac=True,bounds=[(None,None),(.01,10.)],method='L-BFGS-B').x
def aux(d,k):
    cols=[j for j in range(len(d['tasks'])) if j!=k]
    source=np.flatnonzero((d['split']<2)&(d['tid']!=k));known={}
    for i in source:known.setdefault((str(d['canonical'][i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
    a=np.full((len(d['y']),2*len(cols)),np.nan,np.float32)
    for i in np.flatnonzero(d['tid']==k):
        for j,t in enumerate(cols):
            v=known.get((str(d['canonical'][i]),t))
            if v:a[i,j]=np.median(v)
    a[:,len(cols):]=np.isfinite(a[:,:len(cols)])
    return a,source,cols
def load_model(path,seed,trees,depth):
    m=CatBoostClassifier();m.load_model(str(path));p=m.get_all_params()
    assert m.tree_count_==trees and p['random_seed']==seed and p['depth']==depth
    return m
def same(a,b,tol=1e-11):np.testing.assert_allclose(a,b,atol=tol,rtol=tol,equal_nan=True)
def rows(path):return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))
def main():
    d=b.load_data(); release=json.loads((OUT/'release.json').read_text()); assert release['status']=='CANDIDATE_PENDING_REPLAY'
    file=ROOT/release['file'];template=ROOT/'admet_official/submission_template.csv';anchor=ROOT/'admet_submissions/v80_cyp2c9_residual/我从山北来带着兰花草-小分子药物预测技术创新赛-20261023.csv'
    assert b.sha(anchor)==release['anchor_sha256'] and b.sha(file)==release['sha256']
    original=rows(anchor);actual=rows(file);assert len(original)==len(actual)
    records=[];changed=Counter()
    for task in TASKS:
        k=d['tasks'].index(task);folder=OUT/task;cache=np.load(folder/'predictions.npz');meta=json.loads((folder/'metadata.json').read_text())
        tr=np.flatnonzero((d['tid']==k)&(d['split']==0));va=np.flatnonzero((d['tid']==k)&(d['split']==1));fit=np.r_[tr,va];q=np.flatnonzero((d['tid']==k)&(d['split']==2))
        pos=[i for i,r in enumerate(original) if i and r[0]==task];assert len(pos)==len(q)
        base=np.r_[d['proxy_cv'][tr],d['proxy_valid'][va]];qbase=np.array([float(original[i][4]) for i in pos]);a,source,cols=aux(d,k)
        np.testing.assert_array_equal(cache['fit'],fit);np.testing.assert_array_equal(cache['query'],q);np.testing.assert_array_equal(source,np.load(folder/'source_rows.npy'))
        assert k not in cols and np.all(d['tid'][source]!=k);assert meta['target_task_excluded'] and not meta['test_labels_used']
        z,qz=ranklog(base),ranklog(qbase);beta=calibrate(d['y'][fit],z);off=beta[0]+beta[1]*z;qoff=beta[0]+beta[1]*qz;x=np.c_[a[fit],z].astype(np.float32);qx=np.c_[a[q],qz].astype(np.float32)
        same(cache['base_fit'],base);same(cache['base_query'],qbase);same(cache['beta'],beta);same(cache['offset'],off);same(cache['qoff'],qoff);same(cache['x'],x);same(cache['qx'],qx)
        members=[]
        for seed in SEEDS:
            path=folder/f'model_{seed}.cbm';assert b.sha(path)==meta['model_hashes'][path.name];m=load_model(path,seed,meta['trees'],meta['depth']);raw=m.predict(qx,prediction_type='RawFormulaVal');members.append(raw);sample=np.unique([0,len(qx)//2,len(qx)-1]);same(raw[sample],m.predict(qx[sample],prediction_type='RawFormulaVal'))
            records.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=b.sha(path),status='PASS'))
        members=np.stack(members);raw=qoff+members.mean(axis=0);pred=expit(raw);same(cache['residual_members'],members);same(cache['raw'],raw);same(cache['p'],pred)
        exported=np.array([float(actual[i][4]) for i in pos]);same(exported,pred,1e-14)
        records.append(dict(task=task,rows=len(q),status='PASS'))
    for old,new in zip(original[1:],actual[1:]):
        assert old[:4]+old[5:]==new[:4]+new[5:]
        if old[4]!=new[4]:changed[old[0]]+=1
        if old[0] not in TASKS:assert old==new
    assert dict(changed)==release['changed_tasks']=={t:sum(1 for r in original[1:] if r[0]==t) for t in TASKS}
    fmt=verify(file,template);raw=file.read_bytes();ref=template.read_bytes();s=field_spans(raw);t=field_spans(ref);chunks=[];cur=0
    for i in range(1,len(actual)):
        lo,hi=s[i][4];a1,b1=t[i][4];chunks.extend((raw[cur:lo],ref[a1:b1]));cur=hi
    chunks.append(raw[cur:]);assert b''.join(chunks)==ref
    fmt.update(independent_model_reload='PASS',models=9,changed_tasks=dict(changed),unchanged_tasks=18,nonprediction_bytes_exact=True,anchor='v80',prediction_spearman_by_task={task:float(spearmanr([float(r[4]) for r in original[1:] if r[0]==task],[float(r[4]) for r in actual[1:] if r[0]==task]).statistic) for task in TASKS})
    file.with_suffix('.verification.json').write_text(json.dumps(fmt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    result=dict(status='PASS',models=9,records=records,submission=fmt,test_labels_used=False,external_data=False,changed_tasks=dict(changed),unchanged_tasks=18,nonprediction_bytes_exact=True)
    b.save(OUT/'model_verification.json',result);print(json.dumps({'status':'PASS','models':9,'changed_tasks':dict(changed),'sha256':b.sha(file)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
