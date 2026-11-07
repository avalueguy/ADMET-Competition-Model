import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
import sys,time,json,pathlib,numpy as np,joblib
from scipy import sparse
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error as mae
sys.path.insert(0,'/root/autodl-tmp/v76');import admet_v77_ranking as b
D=b.load_data();legacy=np.load(b.ROOT/'recovered_history/legacy_v24_folds.npy')
OUT=b.ROOT/'admet_model_outputs_v80_probe';OUT.mkdir(exist_ok=True)
# exact cached numerical fingerprint views, no new RDKit feature generation
raw=np.load(b.ROOT/'admet_model_outputs_v67/features/official_chemistry.npz')['x']
extra=np.load(b.ROOT/'admet_model_outputs_v73/features/views.npz')['extra']
views=[sparse.csr_matrix(x) for x in [raw[:,:2048],raw[:,2048:4096],extra[:,:2048],extra[:,2048:4096],extra[:,4096:6144],extra[:,6144:]]]
# precompute task kernel all train+valid, retaining full-vs-parent diversity
records=[]
for task in ['caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az']:
 k=D['tasks'].index(task); rows=np.flatnonzero((D['tid']==k)&(D['split']<2));mi=D['mi'][rows];n=len(rows)
 folder=OUT/task;folder.mkdir(exist_ok=True); chemical=np.zeros((n,n),np.float32)
 for v,w in zip(views,[.25,.15,.2,.15,.1,.15]):
  x=v[mi]; norm=np.asarray(x.multiply(x).sum(1)).ravel()
  for st in range(0,n,256):
   dot=(x[st:st+256]@x.T).toarray();tan=dot/np.maximum(norm[st:st+len(dot),None]+norm[None]-dot,1e-8)
   chemical[st:st+len(dot)]+=w*tan
 loc={int(r):j for j,r in enumerate(rows)}
 splits={}
 for fold in [0,1,2]:
  tr=rows[(D['split'][rows]==0)&(legacy[rows]!=fold)];q=rows[(D['split'][rows]==0)&(legacy[rows]==fold)];splits['fold'+str(fold)]=(tr,q)
 splits['valid']=(rows[D['split'][rows]==0],rows[D['split'][rows]==1])
 for part,(tr,q) in splits.items():
  # cached legacy folds: verify no structural overlap
  assert not set(D['groups'][tr])&set(D['groups'][q]); assert not set(D['canonical'][tr])&set(D['canonical'][q])
  a=np.array([loc[int(i)] for i in tr]);z=np.array([loc[int(i)] for i in q]);y=D['y'][tr];vy=D['y'][q]
  med=np.nanmedian(D['views']['physchem'][D['mi'][tr]],axis=0);med=np.nan_to_num(med)
  xx=D['views']['physchem'][D['mi'][rows]];xx=np.where(np.isfinite(xx),xx,med)
  mean=xx[a].mean(0);sd=xx[a].std(0).clip(1e-3);xx=((xx-mean)/sd).clip(-5,5).astype(np.float32)
  norm=(xx*xx).sum(1);R=xx[a];rn=norm[a]
  dist=np.maximum(0,norm[:,None]+rn[None]-2*xx@R.T)/xx.shape[1]
  physical=.5*np.exp(-dist/.5)+.5*np.exp(-dist/2.)
  tf=[float(np.median(y)),max(float(np.quantile(y,.75)-np.quantile(y,.25)),.001)];ty=(y-tf[0])/tf[1]
  old=(D['proxy_cv'] if part!='valid' else D['proxy_valid'])[q]
  for family in [0,1]:
   K=chemical[:,a] if family==0 else .85*chemical[:,a]+.15*physical
   kt=((K[a]+K[a].T)*.5).astype(float);kq=K[z].astype(float)
   for method,val in [('krr',.03),('krr',.1),('krr',.3),('krr',1.),('svr',3.),('svr',10.),('svr',30.)]:
    ident=f'{family}_{method}_{val:g}';dest=folder/part/ident;dest.mkdir(parents=True,exist_ok=True)
    net=KernelRidge(alpha=val,kernel='precomputed') if method=='krr' else SVR(C=val,epsilon=.04,kernel='precomputed')
    net.fit(kt,ty);p=net.predict(kq)*tf[1]+tf[0]
    if task=='ppbr_az':p=p.clip(0,100)
    np.savez_compressed(dest/'predictions.npz',fit=tr,query=q,p=p,baseline=old)
    # save every development model for independent replay
    joblib.dump(dict(model=net,fit=tr,target=tf,family=family,method=method,parameter=val),dest/'model.joblib',compress=3)
    np.savez_compressed(dest/'replay_kernel.npz',x=kq)
    rec=dict(task=task,part=part,family=family,method=method,parameter=val,base=mae(vy,old),expert=mae(vy,p),gains={str(w):float(mae(vy,old)-mae(vy,(1-w)*old+w*p)) for w in [.1,.25,.5,.75,1.]})
    records.append(rec)
  print(json.dumps(dict(task=task,part=part,status='done')),flush=True)
 b.save(OUT/'probe.json',dict(records=records))
# freeze on only folds0/1 then report remaining fold and official valid
choices=[]
for task in ['caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az']:
 grid=[]
 for r in records:
  if r['task']!=task or r['part']!='fold0':continue
  rr=next(v for v in records if all(v[z]==r[z] for z in ['task','family','method','parameter']) and v['part']=='fold1')
  for w in ['0.1','0.25','0.5','0.75','1.0']:
   gs=[r['gains'][w],rr['gains'][w]]
   grid.append(dict(task=task,family=r['family'],method=r['method'],parameter=r['parameter'],weight=float(w),select_gains=gs,objective=min(gs)))
 best=max(grid,key=lambda v:v['objective']);conf=[]
 for part in ['fold2','valid']:
  rr=next(v for v in records if all(v[z]==best[z] for z in ['task','family','method','parameter']) and v['part']==part)
  conf.append(dict(part=part,base=rr['base'],expert=rr['expert'],gain=rr['gains'][str(best['weight'])]))
 best['confirmation']=conf;best['accepted']=best['objective']>0 and all(x['gain']>0 for x in conf);choices.append(best)
 print(json.dumps(best),flush=True)
b.save(OUT/'selection.json',dict(status='COMPLETE',choices=choices,warning='Historical full-pipeline OOF not reconstructed; dev labels reused.'))
