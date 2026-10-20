"""Remote frequency-aware unfolded kernels and cross-task graph phenotype transfer."""
import os
for _k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[_k]='2'
import argparse,csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
import torch,joblib
from scipy import sparse
from sklearn.svm import SVC,SVR
from rdkit import Chem,rdBase
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator,GetMorganFeatureAtomInvGen,GetTopologicalTorsionGenerator
import admet_v24_models as b
import admet_v27_models as graph
import admet_v43_run as graph_source
import admet_v46_run as adapter
import admet_v60_baseline as anchor

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v61'
P=json.loads((ROOT/'admet_v61_protocol.json').read_text(encoding='utf-8'))
PHENO={}

def signature():
    return hashlib.sha256((b.sha(__file__)+b.sha(ROOT/'admet_v61_protocol.json')+graph.signature()).encode()).hexdigest()

def chemistry(d):
    dest=OUT/'features';dest.mkdir(parents=True,exist_ok=True)
    meta=dict(smiles_sha=hashlib.sha256(json.dumps(d['smiles']).encode()).hexdigest(),rdkit=rdBase.rdkitVersion,views=P['chemical_views'],recipe='unfolded_binary_v1')
    path=dest/'chemistry.json'
    names=[x[0] for x in P['chemical_views']]
    if path.exists():
        assert json.loads(path.read_text())==meta
        return [sparse.load_npz(dest/(name+'.npz')) for name in names]
    gens=[GetMorganGenerator(radius=r,includeChirality=True) for r in [1,2,3]]
    gens += [GetMorganGenerator(radius=2,includeChirality=True,atomInvariantsGenerator=GetMorganFeatureAtomInvGen()),GetTopologicalTorsionGenerator(includeChirality=True)]
    vocab=[{} for _ in gens];rr=[[] for _ in gens];cc=[[] for _ in gens]
    for i,sm in enumerate(d['smiles']):
        mol=Chem.MolFromSmiles(sm)
        if mol is None:continue
        for j,gen in enumerate(gens):
            for key in gen.GetSparseCountFingerprint(mol).GetNonzeroElements():
                if key not in vocab[j]:vocab[j][key]=len(vocab[j])
                rr[j].append(i);cc[j].append(vocab[j][key])
        if i%15000==0:b.emit('v61_chemistry',done=i,total=len(d['smiles']))
    result=[]
    for j,name in enumerate(names):
        x=sparse.csr_matrix((np.ones(len(rr[j]),np.float32),(rr[j],cc[j])),shape=(len(d['smiles']),len(vocab[j])))
        sparse.save_npz(dest/(name+'.npz'),x);result.append(x)
    b.save(path,meta);b.save(dest/'feature_audit.json',dict(vocabulary_label_free=True,query_labels_used=False,fixed_length_folding=False,note='RDKit fragment IDs still have intrinsic hashing; no claim of mathematically collision-free IDs.',dimensions=[x.shape[1] for x in result],nonzeros=[x.nnz for x in result]));return result

def phenotype(d,g,partition):
    if partition in PHENO:return PHENO[partition]
    dest=OUT/'features'/partition.replace('/','_');dest.mkdir(parents=True,exist_ok=True);path=dest/'phenotype.npz'
    sources=[graph.OUT/partition/str(seed)/'checkpoint_5500.pt' for seed in P['graph_seeds']]
    meta=dict(graph_signature=graph.signature(),sources=[b.sha(p) for p in sources],readout='mean_graph_hybrid_logits_per_seed')
    if path.exists():
        assert json.loads((dest/'phenotype.json').read_text())==meta
        result=np.load(path)['x'].copy()
    else:
        parts=[]
        for file in sources:
            ck=torch.load(file,map_location='cpu',weights_only=False);assert ck['signature']==graph.signature()
            net=graph.BondNet(*ck['dimensions']).cuda().eval();net.load_state_dict(ck['state'])
            x=adapter.feature_matrix(d,ck,partition);out=[]
            with torch.inference_mode():
                for start in range(0,len(d['smiles']),128):
                    mm=np.arange(start,min(start+128,len(d['smiles'])))
                    out.append(net(graph.pack(g,mm),x[mm]).cpu().numpy().mean(0))
            parts.append(np.concatenate(out));del net;b.emit('v61_phenotype',partition=partition,seed_file=str(file.relative_to(ROOT)))
        result=np.concatenate(parts,1).astype(np.float32);assert np.isfinite(result).all()
        np.savez_compressed(path,x=result);b.save(dest/'phenotype.json',meta)
    PHENO.clear();PHENO[partition]=result;return result

def rbf(a,c):
    z=np.empty((len(a),len(c)),np.float32);norm=(c*c).sum(1)
    for start in range(0,len(a),256):
        x=a[start:start+256];dist=np.maximum(0,(x*x).sum(1)[:,None]+norm[None]-2*x@c.T)/x.shape[1]
        z[start:start+len(x)]=.5*np.exp(-dist/.5)+.5*np.exp(-dist/2)
    return z

def chemical_kernel(views,refs,queries):
    # Global vocabulary uses structures only. IDF uses fitting molecules only.
    # Full query norms retain unseen fragments rather than inflating similarity.
    unique=np.unique(refs);result=np.zeros((len(queries),len(refs)),np.float32)
    for x,(_,weight) in zip(views,P['chemical_views']):
        df=np.asarray(x[unique].sum(0)).ravel();idf=(1+np.log((1+len(unique))/(1+df)))**P['idf_power']
        xx=x.multiply(idf).tocsr();r=xx[refs];q=xx[queries];rn=np.asarray(r.multiply(r).sum(1)).ravel();qn=np.asarray(q.multiply(q).sum(1)).ravel()
        for start in range(0,len(queries),256):
            dot=(q[start:start+256]@r.T).toarray();den=np.maximum(qn[start:start+256,None]+rn[None]-dot,1e-8)
            result[start:start+len(dot)]+=weight*(.7*dot/den+.3*(dot/den)**2)
    return result

def matrices(d,g,views,k,rows,query,partition,chemical_only=False):
    ref=d['mi'][rows];q=d['mi'][query];joint=np.r_[ref,q];n=len(ref)
    ck=chemical_kernel(views,ref,joint)
    raw=np.concatenate([d['desc'],d['emb']],1);pre=b.Preprocessor().fit(raw[np.unique(ref)]);dense=pre.transform(raw[joint]);dk=rbf(dense,dense[:n])
    ppre=None;columns=None;audit={}
    if chemical_only:
        pk=np.zeros_like(ck)
    else:
        audit=graph_source.representation_audit(d,rows,query,partition)
        rawp=phenotype(d,g,partition);nt=len(d['tasks']);columns=np.array([j for j in range(rawp.shape[1]) if j%nt!=k])
        # No target's graph head can dominate this auxiliary phenotype geometry.
        ppre=b.Preprocessor().fit(rawp[np.unique(ref)][:,columns]);px=ppre.transform(rawp[joint][:,columns]);pk=rbf(px,px[:n])
    families=[]
    for a,c,e in P['family_weights']:
        kk=a*ck+c*dk+e*pk;train=kk[:n].astype(np.float64);train=(train+train.T)*.5
        families.append([train,kk[n:].astype(np.float64)])
    return families,dict(dense_pre=pre,phenotype_pre=ppre,phenotype_columns=columns),audit

def fit(d,g,views,k,rows,query,partition,dest,configs=None):
    configs=list(range(len(P['configurations']))) if configs is None else list(configs)
    dest.mkdir(parents=True,exist_ok=True);path=dest/'predictions.npz';sig=signature()
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],rows);np.testing.assert_array_equal(z['query'],query)
        assert set(configs).issubset(z['configs'].tolist());return z['p'].copy()
    assert len(rows) and len(query) and not set(rows)&set(query) and np.all(d['split'][rows]<2)
    if partition!='final':
        assert not set(d['rg'][rows])&set(d['rg'][query])
        assert not set(d['canonical'][d['mi'][rows]])&set(d['canonical'][d['mi'][query]])
    start=time.time();chemical_only=all(P['configurations'][j][0]==0 for j in configs)
    kernels,pre,audit=matrices(d,g,views,k,rows,query,partition,chemical_only)
    m=d['metrics'][k];cls=m in ['AUROC','AUPRC'];tf=b.Target(d['y'][rows],m);y=tf.encode(d['y'][rows])
    _,inv,count=np.unique(d['canonical'][d['mi'][rows]],return_inverse=True,return_counts=True);weight=1/count[inv];weight/=weight.mean()
    pp=np.full((len(P['configurations']),len(query)),np.nan)
    for j in configs:
        family,c=P['configurations'][j];tr,qe=kernels[family]
        model=SVC(C=c,kernel='precomputed',cache_size=256,class_weight=None) if cls else SVR(C=c,epsilon=.04,kernel='precomputed',cache_size=256)
        model.fit(tr,y,sample_weight=weight)
        def predict(mod):
            z=mod.decision_function(qe) if cls else mod.predict(qe)
            return tf.decode(z) if m=='MAE' else z
        p=predict(model);bundle=dict(model=model,target=tf,pre=pre,rows=rows,task=k,config=j,partition=partition,signature=sig)
        file=dest/f'config{j}.joblib';joblib.dump(bundle,file,compress=3)
        np.testing.assert_allclose(p,predict(joblib.load(file)['model']),atol=1e-10,rtol=1e-10);pp[j]=p
    assert np.isfinite(pp[configs]).all();np.savez_compressed(path,p=pp,fit=rows,query=query,signature=sig,configs=configs)
    b.save(dest/'metadata.json',dict(**audit,signature=sig,configs=configs,fit_rows=len(rows),query_rows=len(query),preprocessor_fit_only=True,idf_fit_only=True,own_task_phenotype_removed=True,model_reload_passed=True,test_labels_used=False,local_gpu_training=False,seconds=time.time()-start))
    b.emit('v61_fit',task=d['tasks'][k],partition=dest.name,configs=configs,seconds=round(time.time()-start));return pp

def baseline(d):
    v=np.load(ROOT/'admet_model_outputs_v60/baseline_valid.npy').copy();c=np.load(ROOT/'admet_model_outputs_v60/baseline_cv_proxy.npy').copy();f=np.load(b.OUT/'folds.npy')
    rel=json.loads((ROOT/'admet_model_outputs_v60/release.json').read_text())
    for task,r in rel['selected'].items():
        k=d['tasks'].index(task);m=d['metrics'][k];s=r['selection'];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        mod='admet_model_outputs_v59' if r['family']=='kernel' else 'admet_model_outputs_v58'
        cp=np.load(ROOT/mod/task/'cv_predictions.npy')[s['variant']];vp=np.load(ROOT/mod/task/'valid/predictions.npz')['p']
        if r['family']=='sequence':
            import admet_v58_train as seq
            vp=seq.variants(vp,m)
        v[vi]=b.blend(m,v[vi],vp[s['variant']],s['weight'])
        for fold in range(3):
            ix=tr[f[tr]==fold];c[ix]=b.blend(m,c[ix],cp[ix],s['weight'])
    return v,c

def evaluate(d,g,views,tasks):
    bv,bc=baseline(d);np.save(OUT/'baseline_valid.npy',bv);np.save(OUT/'baseline_cv_proxy.npy',bc);f=np.load(b.OUT/'folds.npy');records=[]
    for task in tasks:
        k=d['tasks'].index(task);m=d['metrics'][k];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1));cp=np.full((6,len(d['y'])),np.nan)
        for fold in range(3):
            q=tr[f[tr]==fold];rr=tr[f[tr]!=fold];cp[:,q]=fit(d,g,views,k,rr,q,'cv/fold'+str(fold),OUT/task/f'fold{fold}')
        np.save(OUT/task/'cv_predictions.npy',cp)
        # Official valid is report-only: no selection or veto can read its metric.
        vp=fit(d,g,views,k,tr,vi,'valid',OUT/task/'valid');grid=[]
        for j in range(6):
            for w in P['blend_weights']:
                gains=[b.delta(m,d['y'][ix],bc[ix],b.blend(m,bc[ix],cp[j,ix],w)) for fold in range(3) for ix in [tr[f[tr]==fold]]]
                obj=float(np.mean(gains)-.10*np.std(gains))
                grid.append(dict(variant=j,weight=w,cv_gains=gains,cv_mean=float(np.mean(gains)),objective=obj,valid_gain=b.delta(m,d['y'][vi],bv[vi],b.blend(m,bv[vi],vp[j],w))))
        held=[]
        for fold in range(3):
            options=[r for r in grid if np.mean([x for j,x in enumerate(r['cv_gains']) if j!=fold])>0]
            chosen=max(options,key=lambda r:np.mean([x for j,x in enumerate(r['cv_gains']) if j!=fold])) if options else None
            held.append(chosen['cv_gains'][fold] if chosen else 0.)
        good=[r for r in grid if r['cv_mean']>P['min_cv_gain'] and r['objective']>P['min_cv_gain'] and sum(x>0 for x in r['cv_gains'])>=2]
        selected=max(good,key=lambda r:r['objective']) if good and np.mean(held)>P['min_held_gain'] else None
        record=dict(task=task,metric=m,selection=selected,held_selection_gains=held,held_mean=float(np.mean(held)),grid=grid,baseline_cv=[b.metric(m,d['y'][ix],bc[ix]) for fold in range(3) for ix in [tr[f[tr]==fold]]])
        records.append(record);b.save(OUT/'analysis.json',dict(status='IN_PROGRESS',signature=signature(),tasks=records,limitations=P['limitations']));b.emit('v61_selection',**{a:v for a,v in record.items() if a!='grid'})
    b.save(OUT/'analysis.json',dict(status='FROZEN',signature=signature(),tasks=records,limitations=P['limitations']));return records

def fresh(d,g,views,records):
    f=np.load(b.OUT/'folds.npy');results=[]
    for r in records:
        s=r['selection']
        if not s:continue
        k=d['tasks'].index(r['task']);m=d['metrics'][k];parts=[]
        for fold in [0,1]:
            q0=np.flatnonzero((d['tid']==k)&(d['split']==0)&(f==fold));groups=np.unique(d['rg'][q0]);seed=P['fresh_seed']+fold+k
            held=np.random.default_rng(seed).permutation(groups)[:max(1,int(.8*len(groups)))];q=q0[np.isin(d['rg'][q0],held)]
            rr=np.flatnonzero((d['tid']==k)&(d['split']<2)&~np.isin(d['rg'],held));partition='cv/fold'+str(fold)
            pp=fit(d,g,views,k,rr,q,partition,OUT/r['task']/f'fresh{fold}',[s['variant']])[s['variant']]
            cq,ctrl=adapter.original(d,k,partition);lookup={int(x):i for i,x in enumerate(cq)};ctrl=ctrl[1,[lookup[int(x)] for x in q]]
            merged=b.blend(m,ctrl,pp,s['weight']);parts.append(dict(fold=fold,fit_rows=len(rr),query_rows=len(q),control=b.metric(m,d['y'][q],ctrl),blend=b.metric(m,d['y'][q],merged),gain=b.delta(m,d['y'][q],ctrl,merged),scaffold_overlap=0,control_is_full_ensemble=False))
        good=bool(np.mean([x['gain'] for x in parts])>.001 and min(x['gain'] for x in parts)>-.03)
        results.append(dict(**r,fresh=parts,fresh_mean=float(np.mean([x['gain'] for x in parts])),pass_gate=good));b.save(OUT/'fresh.json',dict(status='IN_PROGRESS',signature=signature(),tasks=results));b.emit('v61_fresh',task=r['task'],pass_gate=good,results=parts)
    b.save(OUT/'fresh.json',dict(status='COMPLETE',signature=signature(),tasks=results));return results

def export(d,g,views,records):
    source=next((ROOT/'admet_submissions/v60').glob('*.csv'));assert b.sha(source)==P['baseline_sha256'];rows=anchor.rows(source);template=anchor.rows(ROOT/'admet_official/submission_template.csv');replay={};changed={}
    for r in records:
        if not r['pass_gate']:continue
        task=r['task'];k=d['tasks'].index(task);m=d['metrics'][k];s=r['selection'];rr=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2))
        pp=fit(d,g,views,k,rr,q,'final',OUT/task/'final',[s['variant']])[s['variant']]
        pos=[i for i,a in enumerate(rows) if i and a[0]==task];np.testing.assert_array_equal([int(rows[i][1]) for i in pos],d['row'][q]);old=np.array([float(rows[i][4]) for i in pos]);new=b.blend(m,old,pp,s['weight'])
        if task=='ppbr_az':new=new.clip(0,100)
        for i,value in zip(pos,new):rows[i][4]=repr(float(value))
        changed[task]=int(np.sum(new!=old));replay[task]=dict(metric=m,selection=s,baseline=old.tolist(),branch=pp.tolist(),result=new.tolist(),evidence=r)
    if not replay:b.save(OUT/'no_release.json',dict(status='NO_VERIFIED_GAIN',signature=signature()));return
    assert len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,t in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==t[:4]+t[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v61';dest.mkdir(parents=True,exist_ok=True);file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261017.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows);raw=b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8')
    if file.exists():assert file.read_bytes()==raw
    else:file.write_bytes(raw)
    b.save(OUT/'fusion_replay.json',replay);b.save(OUT/'release.json',dict(signature=signature(),selected={r['task']:r for r in records if r['pass_gate']},sha256=b.sha(file),official_score=None,limitations=P['limitations']))
    b.save(dest/'verification.json',dict(status='AWAITING_PLATFORM_EVALUATION',signature=signature(),sha256=b.sha(file),baseline_sha256=b.sha(source),rows=16379,columns=6,tasks=22,changed_tasks=changed,bytes=len(raw),encoding='UTF-8 BOM',line_endings='CRLF',official_score=None,known_score_upper_bound=50.804792,local_gpu_training=False,automatically_uploaded=False));b.emit('v61_export',sha256=b.sha(file),changed_tasks=changed)

def main():
    arg=argparse.ArgumentParser();arg.add_argument('stage',choices=['all','evaluate','fresh','export','smoke']);arg.add_argument('--tasks',nargs='+');a=arg.parse_args();b.remote_only();torch.backends.cuda.matmul.allow_tf32=False;OUT.mkdir(exist_ok=True)
    d=b.build_data();views=chemistry(d);g=graph.graphs(d);tasks=a.tasks or P['task_priority']
    if a.stage=='smoke':
        k=d['tasks'].index(tasks[0]);f=np.load(b.OUT/'folds.npy');tr=np.flatnonzero((d['tid']==k)&(d['split']==0));fit(d,g,views,k,tr[f[tr]!=0],tr[f[tr]==0],'cv/fold0',OUT/tasks[0]/'fold0');return
    if a.stage in ['all','evaluate']:evaluate(d,g,views,tasks)
    if a.stage in ['all','fresh']:fresh(d,g,views,json.loads((OUT/'analysis.json').read_text())['tasks'])
    if a.stage in ['all','export']:export(d,g,views,json.loads((OUT/'fresh.json').read_text())['tasks'])
    b.save(OUT/(a.stage+'_completed.json'),dict(status='COMPLETE',signature=signature()))

if __name__=='__main__':main()
