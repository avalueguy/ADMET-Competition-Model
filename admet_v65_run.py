"""Conditional small-sample foundation experts over official OTHER-assay covariates."""
import os,sys
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'admet_runtime/v63_packages'))
import argparse,json,time,hashlib,gc,csv,io
import numpy as np,torch,joblib
from sklearn.decomposition import PCA
from tabicl import TabICLClassifier,TabICLRegressor
import admet_tabicl_inference as stable
from catboost import CatBoostClassifier,CatBoostRegressor
import admet_v24_models as b
from admet_v31_run import auxiliary
import admet_v61_run as v61
import admet_v60_baseline as anchor
OUT=ROOT/'admet_model_outputs_v65';P=json.loads((ROOT/'admet_v65_protocol.json').read_text(encoding='utf-8'));AUX={}

def signature():return hashlib.sha256((b.sha(__file__)+b.sha(ROOT/'admet_v65_protocol.json')+b.sha(stable.__file__)+b.signature()).encode()).hexdigest()

def design(d,k,rows,query,splits,families):
    key=(k,tuple(splits))
    if key not in AUX:
        AUX.clear();AUX[key]=auxiliary(d,splits,k)
    aux,count,audit=AUX[key];refs=np.unique(d['mi'][rows]);pre=b.Preprocessor().fit(d['desc'][refs]);desc=pre.transform(d['desc']);prea=b.Preprocessor().fit(aux[rows]);aa=prea.transform(aux);xx={};state=dict(desc_pre=pre,aux_pre=prea)
    if 1 in families:
        ep=b.Preprocessor().fit(d['emb'][refs]);emb=ep.transform(d['emb']);nc=min(32,len(refs)-1);pc=PCA(n_components=nc,random_state=P['seed'],svd_solver='randomized').fit(emb[refs]);embedding=pc.transform(emb).astype(np.float32);state.update(embedding_pre=ep,pca=pc)
    for name,rr in [('fit',rows),('query',query)]:
        raw=np.concatenate([desc[d['mi'][rr]],aa[rr]],1);xx[name]={0:raw}
        if 1 in families:xx[name][1]=np.concatenate([raw,embedding[d['mi'][rr]]],1)
    audit.update(conditional_assay_evaluation=True,global_scaffold_evaluation=False,query_other_assay_coverage=float(np.mean(count[query]>0)),target_aux_values_and_mask_removed=True);return xx,state,audit

def fit(d,k,rows,query,splits,dest,families=None):
    families=[0,1] if families is None else list(families);dest.mkdir(parents=True,exist_ok=True);sig=signature();path=dest/'predictions.npz'
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig;np.testing.assert_array_equal(z['fit'],rows);np.testing.assert_array_equal(z['query'],query);assert set(families).issubset(z['families'].tolist());return z['p'].copy()
    assert not set(rows)&set(query) and np.all(d['split'][rows]<2)
    if np.any(d['split'][query]<2):assert not set(d['rg'][rows])&set(d['rg'][query])
    start=time.time();m=d['metrics'][k];cls=m in ['AUROC','AUPRC'];tf=b.Target(d['y'][rows],m);xx,state,audit=design(d,k,rows,query,splits,families);name,sha=P['checkpoint_classifier' if cls else 'checkpoint_regressor'];checkpoint=ROOT/'admet_pretrained/tabicl_v2'/name;assert b.sha(checkpoint)==sha;pp=np.full((2,len(query)),np.nan)
    for j in families:
        model=(TabICLClassifier if cls else TabICLRegressor)(n_estimators=P['n_estimators'],model_path=str(checkpoint),allow_auto_download=False,device='cuda',n_jobs=2,batch_size=1,random_state=P['seed'],kv_cache=False,**stable.options())
        model.fit(xx['fit'][j],tf.encode(d['y'][rows]));p=model.predict_proba(xx['query'][j])[:,1] if cls else model.predict(xx['query'][j],output_type='median' if m=='MAE' else 'mean')
        if m=='MAE':p=tf.decode(p)
        file=dest/f'family{j}.pt';model.save(file,save_model_weights=False,save_training_data=True,save_kv_cache=False);restored=(TabICLClassifier if cls else TabICLRegressor).load(file,device='cuda');q=restored.predict_proba(xx['query'][j])[:,1] if cls else restored.predict(xx['query'][j],output_type='median' if m=='MAE' else 'mean')
        if m=='MAE':q=tf.decode(q)
        np.testing.assert_allclose(p,q,atol=2e-5,rtol=2e-5);pp[j]=p;del model,restored;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(path,p=pp,fit=rows,query=query,signature=sig,families=families);joblib.dump(dict(state=state,target=tf,splits=splits,signature=sig,checkpoint_sha256=sha),dest/'preprocessing.joblib',compress=3);b.save(dest/'metadata.json',dict(**audit,signature=sig,model_reload_passed=True,local_gpu_training=False,test_labels_used=False,seconds=time.time()-start));b.emit('v65_fit',task=d['tasks'][k],partition=dest.name,seconds=round(time.time()-start));return pp

def evaluate(d,tasks):
    bv,bc=v61.baseline(d);f=np.load(b.OUT/'folds.npy');np.save(OUT/'baseline_valid.npy',bv);np.save(OUT/'baseline_cv_proxy.npy',bc);records=[]
    for task in tasks:
        k=d['tasks'].index(task);m=d['metrics'][k];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1));cp=np.full((2,len(d['y'])),np.nan)
        for fold in range(3):
            q=tr[f[tr]==fold];rr=tr[f[tr]!=fold];cp[:,q]=fit(d,k,rr,q,[0],OUT/task/f'fold{fold}')
        np.save(OUT/task/'cv_predictions.npy',cp);vp=fit(d,k,tr,vi,[0],OUT/task/'valid');grid=[]
        for j in range(2):
            for w in P['weights']:
                gains=[b.delta(m,d['y'][ix],bc[ix],b.blend(m,bc[ix],cp[j,ix],w)) for fold in range(3) for ix in [tr[f[tr]==fold]]];grid.append(dict(variant=j,weight=w,cv_gains=gains,cv_mean=float(np.mean(gains)),objective=float(np.mean(gains)-.1*np.std(gains)),valid_gain=b.delta(m,d['y'][vi],bv[vi],b.blend(m,bv[vi],vp[j],w))))
        held=[]
        for fold in range(3):
            eligible=[r for r in grid if np.mean([x for j,x in enumerate(r['cv_gains']) if j!=fold])>.002];s=max(eligible,key=lambda r:np.mean([x for j,x in enumerate(r['cv_gains']) if j!=fold])) if eligible else None;held.append(s['cv_gains'][fold] if s else 0.)
        eligible=[r for r in grid if r['cv_mean']>.004 and r['objective']>.004 and sum(x>0 for x in r['cv_gains'])>=2];s=max(eligible,key=lambda r:r['objective']) if eligible and np.mean(held)>.002 else None;records.append(dict(task=task,metric=m,selection=s,held_gains=held,held_mean=float(np.mean(held)),grid=grid));b.save(OUT/'analysis.json',dict(status='IN_PROGRESS',signature=signature(),tasks=records,limitations=P['limitations']));b.emit('v65_selection',task=task,selection=s,held_mean=float(np.mean(held)))
    b.save(OUT/'analysis.json',dict(status='FROZEN',signature=signature(),tasks=records,limitations=P['limitations']));return records

def fresh(d,records):
    results=[]
    for r in records:
        s=r['selection']
        if not s:continue
        k=d['tasks'].index(r['task']);m=r['metric'];allrows=np.flatnonzero((d['tid']==k)&(d['split']<2));groups=np.unique(d['rg'][allrows]);held=np.random.default_rng(P['fresh_seed']+k).permutation(groups)[:max(1,int(.25*len(groups)))];active=np.isin(d['rg'][allrows],held);rr,q=allrows[~active],allrows[active];pp=fit(d,k,rr,q,[0,1],OUT/r['task']/'fresh',[s['variant']])[s['variant']];xx,state,audit=design(d,k,rr,q,[0,1],[s['variant']]);cls=m in ['AUROC','AUPRC'];tf=b.Target(d['y'][rr],m);model=(CatBoostClassifier if cls else CatBoostRegressor)(iterations=650,depth=5,learning_rate=.04,l2_leaf_reg=15,loss_function='Logloss' if cls else ('MAE' if m=='MAE' else 'RMSE'),task_type='CPU',thread_count=2,verbose=False,allow_writing_files=False,random_seed=P['fresh_seed']);model.fit(xx['fit'][s['variant']],tf.encode(d['y'][rr]));ctrl=model.predict_proba(xx['query'][s['variant']])[:,1] if cls else model.predict(xx['query'][s['variant']]);ctrl=tf.decode(ctrl) if m=='MAE' else ctrl;merged=b.blend(m,ctrl,pp,s['weight']);gain=b.delta(m,d['y'][q],ctrl,merged);results.append(dict(**r,fresh_gain=gain,pass_gate=bool(gain>.001),query_rows=len(q),control='fresh conditional CatBoost, identical feature view and authorized source assays',control_is_full_ensemble=False));b.save(OUT/'fresh.json',dict(status='IN_PROGRESS',signature=signature(),tasks=results));b.emit('v65_fresh',task=r['task'],gain=gain,pass_gate=bool(gain>.001))
    b.save(OUT/'fresh.json',dict(status='COMPLETE',signature=signature(),tasks=results));return results

def export(d,records):
    source=next((ROOT/'admet_submissions/v60').glob('*.csv'));assert b.sha(source)==P['baseline_sha256'];rows=anchor.rows(source);template=anchor.rows(ROOT/'admet_official/submission_template.csv');replay={};changed={}
    for r in records:
        if not r['pass_gate']:continue
        task=r['task'];k=d['tasks'].index(task);m=r['metric'];s=r['selection'];rr=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2));p=fit(d,k,rr,q,[0,1],OUT/task/'final',[s['variant']])[s['variant']];pos=[i for i,a in enumerate(rows) if i and a[0]==task];np.testing.assert_array_equal([int(rows[i][1]) for i in pos],d['row'][q]);old=np.array([float(rows[i][4]) for i in pos]);new=b.blend(m,old,p,s['weight'])
        if task=='ppbr_az':new=new.clip(0,100)
        for i,value in zip(pos,new):rows[i][4]=repr(float(value))
        changed[task]=int(np.sum(new!=old));replay[task]=dict(metric=m,selection=s,baseline=old.tolist(),branch=p.tolist(),result=new.tolist(),evidence=r)
    if not replay:b.save(OUT/'no_release.json',dict(status='NO_VERIFIED_GAIN',signature=signature()));return
    assert len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,t in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==t[:4]+t[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v65';dest.mkdir(parents=True,exist_ok=True);file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261017.csv';buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows);raw=b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8')
    if file.exists():assert file.read_bytes()==raw
    else:file.write_bytes(raw)
    b.save(OUT/'fusion_replay.json',replay);b.save(OUT/'release.json',dict(signature=signature(),selected={r['task']:r for r in records if r['pass_gate']},sha256=b.sha(file),official_score=None,limitations=P['limitations']));b.save(dest/'verification.json',dict(status='AWAITING_PLATFORM_EVALUATION',signature=signature(),sha256=b.sha(file),baseline_sha256=b.sha(source),rows=16379,columns=6,tasks=22,changed_tasks=changed,bytes=len(raw),encoding='UTF-8 BOM',line_endings='CRLF',official_score=None,known_score_upper_bound=50.804792,local_gpu_training=False,automatically_uploaded=False));b.emit('v65_export',sha256=b.sha(file),changed_tasks=changed)

def main():
    arg=argparse.ArgumentParser();arg.add_argument('stage',choices=['all','evaluate','fresh','export']);arg.add_argument('--tasks',nargs='+');a=arg.parse_args();b.remote_only();torch.backends.cuda.matmul.allow_tf32=False;OUT.mkdir(exist_ok=True);d=b.build_data()
    if a.stage in ['all','evaluate']:evaluate(d,a.tasks or P['tasks'])
    if a.stage in ['all','fresh']:fresh(d,json.loads((OUT/'analysis.json').read_text())['tasks'])
    if a.stage in ['all','export']:export(d,json.loads((OUT/'fresh.json').read_text())['tasks'])
    b.save(OUT/(a.stage+'_completed.json'),dict(status='COMPLETE',signature=signature()))

if __name__=='__main__':main()
