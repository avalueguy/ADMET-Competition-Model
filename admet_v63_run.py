"""Remote compact chemical/graph tabular foundation experts with fit-only transforms."""
import os,sys
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'admet_runtime/v63_packages'))
import argparse,json,time,hashlib,csv,io,gc
import numpy as np,torch,joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from tabicl import TabICLClassifier,TabICLRegressor
import admet_tabicl_inference as stable
import admet_v24_models as b
import admet_v43_run as gs
import admet_v46_run as adapter
import admet_v61_run as v61
import admet_v60_baseline as anchor
OUT=ROOT/'admet_model_outputs_v63';P=json.loads((ROOT/'admet_v63_protocol.json').read_text(encoding='utf-8'))

def signature():return hashlib.sha256((b.sha(__file__)+b.sha(ROOT/'admet_v63_protocol.json')+b.sha(stable.__file__)+gs.signature()).encode()).hexdigest()

def pca(raw,refs,components):
    pre=b.Preprocessor().fit(raw[refs]);xx=pre.transform(raw);cols=np.flatnonzero(xx[refs].std(0)>1e-5);nc=min(components,len(refs)-1,len(cols));model=PCA(n_components=nc,svd_solver='randomized',random_state=P['seed']);model.fit(xx[refs][:,cols]);return model.transform(xx[:,cols]).astype(np.float32),dict(pre=pre,cols=cols,pca=model)

def design(d,rows,query,partition,families):
    ref=np.unique(d['mi'][rows]);joint=np.unique(d['mi'][np.r_[rows,query]]);lookup=np.full(len(d['smiles']),-1);lookup[joint]=np.arange(len(joint));pre=b.Preprocessor().fit(d['desc'][ref]);desc=pre.transform(d['desc'][joint]);out={0:desc};state=dict(desc_pre=pre);audit={}
    if 1 in families:
        audit=gs.representation_audit(d,rows,query,partition);graph=gs.representations(d,None,partition);parts=[]
        for name,raw,n in [('graph',graph,64),('embedding',d['emb'],32),('fp',np.concatenate([d['fp2'],d['fcfp']],1),64)]:
            xx,model=pca(raw,ref,n);parts.append(xx[joint]);state[name]=model
        out[1]=np.concatenate([desc,*parts],1)
    return {j:dict(fit=out[j][lookup[d['mi'][rows]]],query=out[j][lookup[d['mi'][query]]]) for j in families},state,audit

def fit(d,k,rows,query,partition,dest,families=None):
    families=[0,1] if families is None else list(families);dest.mkdir(parents=True,exist_ok=True);path=dest/'predictions.npz';sig=signature()
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig;np.testing.assert_array_equal(z['fit'],rows);np.testing.assert_array_equal(z['query'],query);assert set(families).issubset(z['families'].tolist());return z['p'].copy()
    assert not set(rows)&set(query) and np.all(d['split'][rows]<2)
    if partition!='final':assert not set(d['rg'][rows])&set(d['rg'][query])
    start=time.time();m=d['metrics'][k];cls=m in ['AUROC','AUPRC'];tf=b.Target(d['y'][rows],m);xx,state,audit=design(d,rows,query,partition,families);checkpoint,sha=P['checkpoint_files']['classifier' if cls else 'regressor'];modelpath=ROOT/'admet_pretrained/tabicl_v2'/checkpoint;assert b.sha(modelpath)==sha;pp=np.full((2,len(query)),np.nan)
    for j in families:
        model=(TabICLClassifier if cls else TabICLRegressor)(n_estimators=P['ensemble_size'],model_path=str(modelpath),allow_auto_download=False,device='cuda',n_jobs=2,batch_size=1,random_state=P['seed'],kv_cache=False,**stable.options())
        model.fit(xx[j]['fit'],tf.encode(d['y'][rows]));p=model.predict_proba(xx[j]['query'])[:,1] if cls else model.predict(xx[j]['query'],output_type='median' if m=='MAE' else 'mean')
        if m=='MAE':p=tf.decode(p)
        file=dest/f'family{j}.pt';model.save(file,save_model_weights=False,save_training_data=True,save_kv_cache=False)
        restored=(TabICLClassifier if cls else TabICLRegressor).load(file,device='cuda');z=restored.predict_proba(xx[j]['query'])[:,1] if cls else restored.predict(xx[j]['query'],output_type='median' if m=='MAE' else 'mean')
        if m=='MAE':z=tf.decode(z)
        np.testing.assert_allclose(p,z,atol=2e-5,rtol=2e-5);pp[j]=p;del model,restored;gc.collect();torch.cuda.empty_cache()
    assert np.isfinite(pp[families]).all();joblib.dump(dict(state=state,target=tf,signature=sig,rows=rows,query=query,checkpoint_sha256=sha),dest/'preprocessing.joblib',compress=3);np.savez_compressed(path,p=pp,fit=rows,query=query,signature=sig,families=families)
    b.save(dest/'metadata.json',dict(**audit,signature=sig,fit_rows=len(rows),query_rows=len(query),families=families,model_reload_passed=True,fit_only_pca=True,test_labels_used=False,local_gpu_training=False,seconds=time.time()-start));b.emit('v63_fit',task=d['tasks'][k],partition=dest.name,families=families,seconds=round(time.time()-start));return pp

def evaluate(d,tasks):
    bv,bc=v61.baseline(d);f=np.load(b.OUT/'folds.npy');np.save(OUT/'baseline_valid.npy',bv);np.save(OUT/'baseline_cv_proxy.npy',bc);records=[]
    for task in tasks:
        k=d['tasks'].index(task);m=d['metrics'][k];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1));cp=np.full((2,len(d['y'])),np.nan)
        for fold in range(3):
            q=tr[f[tr]==fold];rr=tr[f[tr]!=fold];cp[:,q]=fit(d,k,rr,q,'cv/fold'+str(fold),OUT/task/f'fold{fold}')
        np.save(OUT/task/'cv_predictions.npy',cp);vp=fit(d,k,tr,vi,'valid',OUT/task/'valid');grid=[]
        for j in range(2):
            for w in P['weights']:
                gains=[b.delta(m,d['y'][ix],bc[ix],b.blend(m,bc[ix],cp[j,ix],w)) for fold in range(3) for ix in [tr[f[tr]==fold]]];grid.append(dict(variant=j,weight=w,cv_gains=gains,cv_mean=float(np.mean(gains)),objective=float(np.mean(gains)-.1*np.std(gains)),valid_gain=b.delta(m,d['y'][vi],bv[vi],b.blend(m,bv[vi],vp[j],w))))
        held=[]
        for fold in range(3):
            eligible=[r for r in grid if np.mean([x for j,x in enumerate(r['cv_gains']) if j!=fold])>.002];s=max(eligible,key=lambda r:np.mean([x for j,x in enumerate(r['cv_gains']) if j!=fold])) if eligible else None;held.append(s['cv_gains'][fold] if s else 0.)
        eligible=[r for r in grid if r['cv_mean']>.003 and r['objective']>.003 and sum(x>0 for x in r['cv_gains'])>=2];s=max(eligible,key=lambda r:r['objective']) if eligible and np.mean(held)>.002 else None;records.append(dict(task=task,metric=m,selection=s,held_gains=held,held_mean=float(np.mean(held)),grid=grid));b.save(OUT/'analysis.json',dict(status='IN_PROGRESS',signature=signature(),tasks=records,limitations=P['limitations']));b.emit('v63_selection',task=task,selection=s,held_mean=float(np.mean(held)))
    b.save(OUT/'analysis.json',dict(status='FROZEN',signature=signature(),tasks=records,limitations=P['limitations']));return records

def fresh(d,records):
    f=np.load(b.OUT/'folds.npy');results=[]
    for r in records:
        s=r['selection']
        if not s:continue
        k=d['tasks'].index(r['task']);m=r['metric'];parts=[]
        for fold in [0,1]:
            q0=np.flatnonzero((d['tid']==k)&(d['split']==0)&(f==fold));groups=np.unique(d['rg'][q0]);held=np.random.default_rng(P['fresh_seed']+fold+k).permutation(groups)[:max(1,int(.8*len(groups)))];q=q0[np.isin(d['rg'][q0],held)];rr=np.flatnonzero((d['tid']==k)&(d['split']<2)&~np.isin(d['rg'],held));partition='cv/fold'+str(fold);pp=fit(d,k,rr,q,partition,OUT/r['task']/f'fresh{fold}',[s['variant']])[s['variant']];cq,ctrl=adapter.original(d,k,partition);lookup={int(x):i for i,x in enumerate(cq)};ctrl=ctrl[1,[lookup[int(x)] for x in q]];merged=b.blend(m,ctrl,pp,s['weight']);parts.append(dict(gain=b.delta(m,d['y'][q],ctrl,merged),control=b.metric(m,d['y'][q],ctrl),blend=b.metric(m,d['y'][q],merged),rows=len(q),control_is_full_ensemble=False))
        if s['variant']==0:
            allrows=np.flatnonzero((d['tid']==k)&(d['split']<2));groups=np.unique(d['rg'][allrows]);held=np.random.default_rng(P['fresh_seed']+100+k).permutation(groups)[:max(1,int(.25*len(groups)))];mask=np.isin(d['rg'][allrows],held);rr,q=allrows[~mask],allrows[mask];pp=fit(d,k,rr,q,'fresh',OUT/r['task']/'fresh_independent',[0])[0];xx,_,_=design(d,rr,q,'fresh',[0]);cls=m in ['AUROC','AUPRC'];tf=b.Target(d['y'][rr],m);ctrl=(ExtraTreesClassifier if cls else ExtraTreesRegressor)(n_estimators=300,min_samples_leaf=2,max_features=.8,random_state=P['fresh_seed'],n_jobs=2);ctrl.fit(xx[0]['fit'],tf.encode(d['y'][rr]));p=ctrl.predict_proba(xx[0]['query'])[:,1] if cls else ctrl.predict(xx[0]['query']);p=tf.decode(p) if m=='MAE' else p;merged=b.blend(m,p,pp,s['weight']);parts.append(dict(gain=b.delta(m,d['y'][q],p,merged),rows=len(q),control='fresh ExtraTrees',control_is_full_ensemble=False))
        good=bool(np.mean([x['gain'] for x in parts])>.001 and min(x['gain'] for x in parts)>-.03);results.append(dict(**r,fresh=parts,fresh_mean=float(np.mean([x['gain'] for x in parts])),pass_gate=good));b.save(OUT/'fresh.json',dict(status='IN_PROGRESS',signature=signature(),tasks=results));b.emit('v63_fresh',task=r['task'],pass_gate=good,results=parts)
    b.save(OUT/'fresh.json',dict(status='COMPLETE',signature=signature(),tasks=results));return results

def export(d,records):
    source=next((ROOT/'admet_submissions/v60').glob('*.csv'));assert b.sha(source)==P['baseline_sha256'];rows=anchor.rows(source);template=anchor.rows(ROOT/'admet_official/submission_template.csv');replay={};changed={}
    for r in records:
        if not r['pass_gate']:continue
        task=r['task'];k=d['tasks'].index(task);m=r['metric'];s=r['selection'];rr=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2));p=fit(d,k,rr,q,'final',OUT/task/'final',[s['variant']])[s['variant']];pos=[i for i,a in enumerate(rows) if i and a[0]==task];np.testing.assert_array_equal([int(rows[i][1]) for i in pos],d['row'][q]);old=np.array([float(rows[i][4]) for i in pos]);new=b.blend(m,old,p,s['weight'])
        if task=='ppbr_az':new=new.clip(0,100)
        for i,value in zip(pos,new):rows[i][4]=repr(float(value))
        changed[task]=int(np.sum(new!=old));replay[task]=dict(metric=m,selection=s,baseline=old.tolist(),branch=p.tolist(),result=new.tolist(),evidence=r)
    if not replay:b.save(OUT/'no_release.json',dict(status='NO_VERIFIED_GAIN',signature=signature()));return
    assert len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,t in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==t[:4]+t[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v63';dest.mkdir(parents=True,exist_ok=True);file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261017.csv';buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows);raw=b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8')
    if file.exists():assert file.read_bytes()==raw
    else:file.write_bytes(raw)
    b.save(OUT/'fusion_replay.json',replay);b.save(OUT/'release.json',dict(signature=signature(),selected={r['task']:r for r in records if r['pass_gate']},sha256=b.sha(file),official_score=None,limitations=P['limitations']));b.save(dest/'verification.json',dict(status='AWAITING_PLATFORM_EVALUATION',signature=signature(),sha256=b.sha(file),baseline_sha256=b.sha(source),rows=16379,columns=6,tasks=22,changed_tasks=changed,bytes=len(raw),encoding='UTF-8 BOM',line_endings='CRLF',official_score=None,known_score_upper_bound=50.804792,local_gpu_training=False,automatically_uploaded=False));b.emit('v63_export',sha256=b.sha(file),changed_tasks=changed)

def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['all','evaluate','fresh','export']);a.add_argument('--tasks',nargs='+');args=a.parse_args();b.remote_only();torch.backends.cuda.matmul.allow_tf32=False;OUT.mkdir(exist_ok=True);d=b.build_data()
    if args.stage in ['all','evaluate']:evaluate(d,args.tasks or P['tasks'])
    if args.stage in ['all','fresh']:fresh(d,json.loads((OUT/'analysis.json').read_text())['tasks'])
    if args.stage in ['all','export']:export(d,json.loads((OUT/'fresh.json').read_text())['tasks'])
    b.save(OUT/(args.stage+'_completed.json'),dict(status='COMPLETE',signature=signature()))

if __name__=='__main__':main()
