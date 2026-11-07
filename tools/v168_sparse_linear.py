"""Fit-only TF-IDF molecular fragments and regularized linear task experts."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import hashlib,json,sys,time
from pathlib import Path
import joblib,numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.preprocessing import normalize
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v147_path_features as paths
import v104_sparse_features as circles
import v148_experiment_runner as runner
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v168'
CONFIGS=(dict(strength=1.),dict(strength=10.))
METHOD='Independent sparse TF-IDF linear fragment experts, five unfolded circular/atom-pair/torsion/path views with equal block normalization; augmented chemistry and target-excluded official context. Fit-only prevalence, IDF and descriptor scaling. Classification uses fit-label NB log-ratios and L2 logistic regression; continuous outcomes use ridge on native MAE or rank target. No historical predictions as features.'

def sig():return hashlib.sha256((b.sha(__file__)+v.sig()+paths.sig()+circles.sig()+b.sha(runner.__file__)).encode()).hexdigest()
def key(c):return 'strength'+str(c['strength'])

def load():
    d=paths.load(v.load());audit=json.loads((circles.OUT/'audit.json').read_text());assert audit['schema']['signature']==circles.sig()
    raw=[]
    for j in range(2):
        p=circles.OUT/f'view{j}.npz';assert b.sha(p)==audit['views'][j]['sha256'];raw.append(sparse.load_npz(p))
    d['v168_circular']=raw;return d

def design(d,k,part):
    fit,q,x,qx,source=v.old.features(d,k,part,'augmented');cls=d['metrics'][k] in ('AUROC','AUPRC');blocks=[];queries=[];states=[]
    for raw in d['v168_circular']+d['path_fingerprints']:
        a=raw[d['mi'][fit]].astype(float);z=raw[d['mi'][q]].astype(float);freq=np.asarray((a>0).sum(0)).ravel();cols=np.flatnonzero(freq>=3)
        if len(cols)>60000:cols=np.sort(cols[np.argsort(freq[cols],kind='stable')[-60000:]])
        a,z=a[:,cols],z[:,cols];a.data=np.log1p(a.data);z.data=np.log1p(z.data);idf=np.log((1+len(fit))/(1+freq[cols]))+1
        ratio=np.ones(len(cols))
        if cls:
            y=d['y'][fit];positive=np.asarray((a[y>.5]>0).sum(0)).ravel()+1;negative=np.asarray((a[y<.5]>0).sum(0)).ravel()+1
            ratio=np.log(positive/positive.sum())-np.log(negative/negative.sum())
        a=normalize(a.multiply(idf*ratio).tocsr(),norm='l2');z=normalize(z.multiply(idf*ratio).tocsr(),norm='l2')
        blocks.append(a/np.sqrt(5));queries.append(z/np.sqrt(5));states.append(dict(columns=cols,idf=idf,ratio=ratio))
    # Work in float64 before estimating scale to prevent float32 overflow.
    x,qx=x.astype(float),qx.astype(float);median=np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0. for z in x.T]);a=np.where(np.isfinite(x),x,median);center=a.mean(0);scale=a.std(0).clip(.05)
    zz=np.where(np.isfinite(qx),qx,median);width=x.shape[1];a=np.c_[np.clip((a-center)/scale,-5,5),np.isfinite(x)]/np.sqrt(width);zz=np.c_[np.clip((zz-center)/scale,-5,5),np.isfinite(qx)]/np.sqrt(width)
    blocks.append(sparse.csr_matrix(a));queries.append(sparse.csr_matrix(zz));xx=sparse.hstack(blocks,format='csr');qq=sparse.hstack(queries,format='csr')
    assert np.isfinite(xx.data).all() and np.isfinite(qq.data).all() and not set(fit)&set(q) and np.all(d['tid'][source]!=k)
    return dict(fit=fit,query=q,x=xx,qx=qq,source=source,states=states,median=median,center=center,scale=scale)

def predict(d,k,part,c):
    folder=OUT/'models'/d['tasks'][k]/part/key(c);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);return z['p'].copy()
    a=design(d,k,part);y=d['y'][a['fit']];m=d['metrics'][k];cls=m in ('AUROC','AUPRC');center=float(np.median(y)) if m=='MAE' else 0.;scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if m=='MAE' else 1.
    target=y if cls else r83.ranknorm(y) if m=='Spearman' else (y-center)/scale
    model=LogisticRegression(C=c['strength'],solver='liblinear',max_iter=3000,tol=1e-5,random_state=1684026) if cls else Ridge(alpha=c['strength'],solver='lsqr',tol=1e-6,max_iter=3000)
    start=time.monotonic();model.fit(a['x'],target)
    if cls:assert model.n_iter_.max()<3000
    raw=model.decision_function(a['qx']) if cls else model.predict(a['qx']);p=expit(raw) if cls else raw*scale+center
    if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
    folder.mkdir(parents=True,exist_ok=True);mp=folder/'model.joblib';joblib.dump(model,mp,compress=3);joblib.dump({key:value for key,value in a.items() if key not in ('x','qx')},folder/'transform.joblib',compress=3)
    replay=joblib.load(mp);np.testing.assert_array_equal(raw,replay.decision_function(a['qx']) if cls else replay.predict(a['qx']))
    np.savez_compressed(file,signature=sig(),fit=a['fit'],query=a['query'],source=a['source'],p=p,center=center,scale=scale)
    print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,config=c,columns=a['x'].shape[1],seconds=round(time.monotonic()-start))),flush=True);return p

def smoke(d):
    from v161_verify_and_assemble import same
    checks=[]
    for task in ('herg','caco2_wang','clearance_hepatocyte_az'):
        k=d['tasks'].index(task);a=design(d,k,'fold0');q=a['query'];saved=d['y'][q].copy()
        try:
            d['y'][q]=987654.;d.pop('_v96_assays',None);z=design(d,k,'fold0');same(a,z)
        finally:d['y'][q]=saved;d.pop('_v96_assays',None)
        checks.append(dict(task=task,query_labels_not_in_features=True,fit_preprocessing_finite=True,columns=a['x'].shape[1]))
    b.save(OUT/'input_audit.json',dict(status='PASS',signature=sig(),checks=checks,test_labels_used=False));print('INPUT_AUDIT_PASS',flush=True)

def main():
    d=load();OUT.mkdir(exist_ok=True);smoke(d);runner.run(sys.modules[__name__],d)
if __name__=='__main__':main()
