"""Train and export the v81 three task residual candidate on the GPU host."""
from __future__ import annotations
import csv, io, json, hashlib, sys
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata
from scipy.optimize import minimize
from catboost import CatBoostClassifier, Pool

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import admet_v77_ranking as b
from repack_upload_template import field_spans
from verify_submission import verify

TASKS=('bioavailability_ma','cyp2d6_veith','pgp_broccatelli')
SEEDS=(802026,802027,802028)
CFG={'bioavailability_ma':(False,3,100),'cyp2d6_veith':(False,4,300),'pgp_broccatelli':(False,3,100)}
ANCHOR=ROOT/'admet_submissions/v80_cyp2c9_residual/我从山北来带着兰花草-小分子药物预测技术创新赛-20261023.csv'
ANCHOR_SHA='fd36df4fa66bd5619df228a9748ab57f34e0776c60fb89b301b2f10961036538'
OUT=ROOT/'admet_model_outputs_v81_final'

def ranklog(x): return logit(np.clip((rankdata(np.asarray(x))-.5)/len(x),.002,.998))
def calibrate(y,z):
    x=np.c_[np.ones(len(z)),z]
    def fun(t):
        eta=x@t;r=expit(eta)-y
        return (np.logaddexp(0,eta)-y*eta).mean()+.001*t[1]**2, x.T@r/len(r)+[0,.002*t[1]]
    return minimize(fun,[0.,1.],jac=True,bounds=[(None,None),(.01,10.)],method='L-BFGS-B').x
def aux(d,k):
    cols=[j for j in range(len(d['tasks'])) if j!=k]
    source=np.flatnonzero((d['split']<2)&(d['tid']!=k)); known={}
    for i in source: known.setdefault((str(d['canonical'][i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
    a=np.full((len(d['y']),2*len(cols)),np.nan,np.float32)
    for i in np.flatnonzero(d['tid']==k):
        for j,t in enumerate(cols):
            v=known.get((str(d['canonical'][i]),t))
            if v: a[i,j]=np.median(v)
    a[:,len(cols):]=np.isfinite(a[:,:len(cols)])
    return a,source,cols
def main():
    d=b.load_data(); assert b.sha(ANCHOR)==ANCHOR_SHA
    rows=list(csv.reader(io.StringIO(ANCHOR.read_bytes().decode('utf-8-sig'),newline='')))
    changed={}; source_counts={}; task_meta={}
    for task in TASKS:
        k=d['tasks'].index(task); _,depth,trees=CFG[task][0],CFG[task][1],CFG[task][2]
        tr=np.flatnonzero((d['tid']==k)&(d['split']==0));va=np.flatnonzero((d['tid']==k)&(d['split']==1));fit=np.r_[tr,va];q=np.flatnonzero((d['tid']==k)&(d['split']==2))
        pos=[i for i,r in enumerate(rows) if i and r[0]==task];assert len(pos)==len(q)
        base=np.r_[d['proxy_cv'][tr],d['proxy_valid'][va]];qbase=np.array([float(rows[i][4]) for i in pos])
        a,source,cols=aux(d,k); z,qz=ranklog(base),ranklog(qbase); beta=calibrate(d['y'][fit],z); off=beta[0]+beta[1]*z;qoff=beta[0]+beta[1]*qz
        x=np.c_[a[fit],z].astype(np.float32);qx=np.c_[a[q],qz].astype(np.float32)
        folder=OUT/task;folder.mkdir(parents=True,exist_ok=True);members=[]; hashes={}
        for seed in SEEDS:
            cfg=dict(iterations=trees,depth=depth,learning_rate=.025,l2_leaf_reg=50,loss_function='Logloss',random_seed=seed,thread_count=8,verbose=False,allow_writing_files=False,bootstrap_type='Bernoulli',subsample=.85,task_type='GPU',devices='0')
            model=CatBoostClassifier(**cfg);model.fit(Pool(x,d['y'][fit],baseline=off));members.append(model.predict(qx,prediction_type='RawFormulaVal'))
            mp=folder/f'model_{seed}.cbm';model.save_model(str(mp));hashes[mp.name]=b.sha(mp)
        members=np.stack(members);pred=expit(qoff+members.mean(axis=0))
        np.savez_compressed(folder/'predictions.npz',fit=fit,query=q,base_fit=base,base_query=qbase,beta=beta,x=x,qx=qx,offset=off,qoff=qoff,residual_members=members,raw=qoff+members.mean(axis=0),p=pred)
        np.save(folder/'source_rows.npy',source)
        b.save(folder/'metadata.json',dict(task=task,seeds=list(SEEDS),trees=trees,depth=depth,model_hashes=hashes,fit_rows=len(fit),query_rows=len(q),source_rows=len(source),source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),target_task_excluded=True,state_sha256=b.sha(folder/'predictions.npz'),test_labels_used=False,external_data=False))
        for i,v in zip(pos,pred): rows[i][4]=repr(float(v))
        changed[task]=len(pos);source_counts[task]=len(source);task_meta[task]=dict(depth=depth,trees=trees)
    template=ROOT/'admet_official/submission_template.csv'; raw=template.read_bytes();spans=field_spans(raw);chunks=[];cur=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=spans[i][4]; value=row[4].encode('ascii'); assert not any(c in value for c in b',\r\n"');chunks.extend((raw[cur:lo],value));cur=hi
    chunks.append(raw[cur:]);dest=ROOT/'admet_submissions/v81_multitask_residual';dest.mkdir(parents=True,exist_ok=True);file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261024.csv';file.write_bytes(b''.join(chunks))
    audit=verify(file,template);audit.update(anchor='v80',anchor_sha256=ANCHOR_SHA,changed_tasks=changed,unchanged_tasks=18,source_rows_by_task=source_counts,models=9,final_models=9,test_labels_used=False,external_data=False)
    file.with_suffix('.verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    b.save(OUT/'release.json',dict(status='CANDIDATE_PENDING_REPLAY',version='v81',file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),anchor='v80',anchor_sha256=ANCHOR_SHA,changed_tasks=changed,unchanged_tasks=18,models=9,final_models=9,config=task_meta,sources=source_counts,official_score=None,test_labels_used=False,external_data=False,verification=audit))
    print(json.dumps(dict(status='EXPORTED',file=str(file),sha256=b.sha(file),changed_tasks=changed),ensure_ascii=False),flush=True)
if __name__=='__main__': main()
