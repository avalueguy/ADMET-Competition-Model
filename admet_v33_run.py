"""Remote learned transfer from experimental assays of chemical neighbors."""
import argparse, csv, hashlib, io, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from catboost import CatBoostClassifier, CatBoostRegressor
import admet_v24_models as b
from admet_v31_run import auxiliary
from admet_v32_run import variants as v32_variants

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_model_outputs_v33'
P=json.loads((ROOT/'admet_v33_protocol.json').read_text(encoding='utf-8'))

def signature():
    names=['admet_v33_run.py','admet_v33_protocol.json','admet_v32_run.py',
        'admet_model_outputs_v32/analysis.json','admet_v31_run.py']
    return hashlib.sha256(json.dumps(dict(base=b.signature(),files={n:b.sha(ROOT/n) for n in names}),sort_keys=True).encode()).hexdigest()

def baseline(d):
    report=json.loads((ROOT/'admet_model_outputs_v32/analysis.json').read_text())
    valid=np.load(ROOT/'admet_model_outputs_v32/baseline_valid.npy').copy()
    cv=np.load(ROOT/'admet_model_outputs_v32/baseline_cv_proxy.npy').copy()
    f=np.load(b.OUT/'folds.npy')
    for r in report['tasks']:
        s=r['selection']
        if s is None:continue
        k=d['tasks'].index(r['task']);m=r['metric']
        vi=np.flatnonzero((d['tid']==k)&(d['split']==1));tr=np.flatnonzero((d['tid']==k)&(f>=0))
        z=np.load(ROOT/'admet_model_outputs_v32'/r['task']/'valid/predictions.npz')
        np.testing.assert_array_equal(z['query'],vi)
        v=v32_variants(z['p'],m)[s['variant']]
        gain=b.delta(m,d['y'][vi],valid[vi],b.blend(m,valid[vi],v,s['weight']))
        np.testing.assert_allclose(gain,s['valid_gain'],rtol=1e-8,atol=1e-10)
        valid[vi]=b.blend(m,valid[vi],v,s['weight'])
        c=np.load(ROOT/'admet_model_outputs_v32'/r['task']/'cv_predictions.npy')[s['variant']]
        for fold in range(3):
            ix=tr[f[tr]==fold];cv[ix]=b.blend(m,cv[ix],c[ix],s['weight'])
    np.save(OUT/'baseline_valid.npy',valid);np.save(OUT/'baseline_cv_proxy.npy',cv)
    return valid,cv

def aggregate(d,rows):
    """One value per canonical identity, so repeated assays cannot dominate retrieval."""
    assert len(rows) and np.all(d['split'][rows]<2)
    k=int(d['tid'][rows[0]]);assert np.all(d['tid'][rows]==k)
    can=d['canonical'][d['mi'][rows]];unique,inverse=np.unique(can,return_inverse=True)
    refs=[];values=[]
    for j in range(len(unique)):
        ix=rows[inverse==j];refs.append(d['mi'][ix[0]])
        yy=d['y'][ix]
        values.append(np.mean(yy) if d['metrics'][k] in ['AUROC','AUPRC'] else np.median(yy))
    return np.array(refs,int),np.array(values,np.float32)

_RETRIEVAL={}
def retrieval_state(d):
    key=id(d['fp2'])
    if _RETRIEVAL.get('key')!=key:
        _RETRIEVAL.clear();_RETRIEVAL['key']=key
        for name in ['fp2','fcfp']:
            fp=torch.tensor((d[name]>0).astype(np.float32),device='cuda')
            _RETRIEVAL[name]=fp;_RETRIEVAL[name+'_sum']=fp.sum(1)
        _,_RETRIEVAL['canonical_id']=np.unique(d['canonical'],return_inverse=True)
        _,_RETRIEVAL['group_id']=np.unique(d['groups'],return_inverse=True)
    return _RETRIEVAL

def similarity(d,q,r):
    state=retrieval_state(d)
    result=None
    for name,w in [('fp2',.65),('fcfp',.35)]:
        a=state[name][q];bb=state[name][r]
        dot=a@bb.T;v=w*dot/(state[name+'_sum'][q,None]+state[name+'_sum'][r][None]-dot).clamp_min(1)
        result=v if result is None else result+v
    return result

def retrieve(d,queries,refs,values,exclude_scaffold):
    """Six summaries, with zero weight for prohibited or empty neighbors."""
    result=np.zeros((len(queries),6),np.float32);count=min(P['neighbors'],len(refs))
    state=retrieval_state(d);canonical=state['canonical_id'];groups=state['group_id']
    # Fingerprints are label-free; values always come from explicit authorized rows.
    for start in range(0,len(queries),256):
        q=queries[start:start+256];sim=similarity(d,q,refs)
        prohibited=(groups[q,None]==groups[refs][None]) if exclude_scaffold else (canonical[q,None]==canonical[refs][None])
        sim.masked_fill_(torch.tensor(prohibited,device='cuda'),-1.)
        scores,ix=sim.topk(count,dim=1);s=scores.cpu().numpy();n=ix.cpu().numpy()
        yy=values[n];ok=s>=0;s=np.maximum(s,0)
        for number,power,column in [(8,4.,0),(48,2.,1)]:
            width=min(number,count);w=np.power(s[:,:width],power)*ok[:,:width]
            den=w.sum(1);mu=(w*yy[:,:width]).sum(1)/np.maximum(den,1e-12)
            result[start:start+len(q),column]=mu
            if column==1:
                var=(w*(yy[:,:width]-mu[:,None])**2).sum(1)/np.maximum(den,1e-12)
                result[start:start+len(q),4]=np.sqrt(np.maximum(var,0))
                result[start:start+len(q),5]=den**2/np.maximum((w*w).sum(1),1e-12)/width
        result[start:start+len(q),2]=s[:,0]
        result[start:start+len(q),3]=s.mean(1)
        del sim,scores,ix
    assert np.isfinite(result).all()
    return result

def other_cache(d,splits):
    """Compute all task summaries once, then remove the target block per experiment."""
    path=OUT/'features'/('other_'+''.join(map(str,splits))+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        z=np.load(path);assert str(z['signature'])==signature();return z['x'].copy()
    q=np.arange(len(d['smiles']));out=np.zeros((len(q),len(d['tasks']),6),np.float32)
    for k,t in enumerate(d['tasks']):
        rows=np.flatnonzero((d['tid']==k)&np.isin(d['split'],splits))
        refs,y=aggregate(d,rows);tf=b.Target(y,d['metrics'][k])
        out[:,k]=retrieve(d,q,refs,tf.encode(y),False)
        b.emit('v33_retrieval_cache',task=t,splits=splits,reference_identities=len(refs))
    np.savez_compressed(path,x=out,signature=signature());torch.cuda.empty_cache();return out

def features(d,k,rows,query,splits,other):
    assert not set(rows)&set(query) and np.all(d['tid'][rows]==k) and np.all(d['split'][rows]<2)
    if np.any(d['split'][query]<2):assert not set(d['rg'][rows])&set(d['rg'][query])
    exact,coverage,audit=auxiliary(d,splits,k)
    columns=[j for j in range(len(d['tasks'])) if j!=k]
    neighbor=other[:,columns].reshape(len(d['smiles']),-1)
    dense=np.concatenate([d['desc'],d['emb']],1);pre=b.Preprocessor().fit(dense[np.unique(d['mi'][rows])]);dense=pre.transform(dense)
    fps=np.concatenate([d['fp2'],d['fcfp']],1)
    variance=np.var(fps[np.unique(d['mi'][rows])],0)
    chosen=np.argsort(variance,kind='stable')[-P['fragment_columns']:]
    refs,y=aggregate(d,rows);tf=b.Target(y,d['metrics'][k]);own_y=tf.encode(y)
    unique=np.unique(d['mi'][np.r_[rows,query]])
    target=np.zeros((len(d['smiles']),6),np.float32);target[unique]=retrieve(d,unique,refs,own_y,True)
    parts={}
    for name,rr in [('fit',rows),('query',query)]:
        mol=d['mi'][rr];aux=np.concatenate([exact[rr],neighbor[mol],target[mol]],1)
        trees=np.concatenate([dense[mol],aux],1)
        parts[name]=dict(fp=fps[mol],dense=dense[mol],aux=aux,
            trees=[trees,np.concatenate([trees,fps[mol][:,chosen]],1)])
    audit.update(other_neighbor_target_block_removed=True,own_neighbors_same_scaffold_excluded=True,
        own_neighbor_source_rows=rows.tolist(),fragment_columns=chosen.tolist(),
        query_exact_assay_coverage=float(np.mean(coverage[query]>0)),retrieved_other_tasks=len(columns),
        query_own_cross_scaffold_similarity_quantiles=np.quantile(target[d['mi'][query],2],[0,.1,.5,.9,1]).tolist(),
        query_own_cross_scaffold_distant_fraction=float(np.mean(target[d['mi'][query],2]<.3)))
    return parts,pre,chosen,tf,audit

class RetrievalNet(nn.Module):
    def __init__(self,dims):
        super().__init__();h=P['hidden'];drop=P['dropout']
        def layer(a,z):return nn.Sequential(nn.Linear(a,z),nn.LayerNorm(z),nn.SiLU(),nn.Dropout(drop))
        self.fp=layer(dims[0],h);self.dense=layer(dims[1],h);self.aux=layer(dims[2],h)
        self.gate=nn.Sequential(nn.Linear(h*3,64),nn.SiLU(),nn.Linear(64,3))
        self.residual=layer(h,h);self.head=nn.Linear(h,1)
    def forward(self,fp,dense,aux):
        if self.training:
            # Drop the full evidence channel per molecule, making missing-neighbor cases trainable.
            aux=aux*(torch.rand((len(aux),1),device=aux.device)>.12)
        h=torch.stack([self.fp(fp),self.dense(dense),self.aux(aux)],1)
        w=self.gate(h.flatten(1)).softmax(1);z=(h*w.unsqueeze(-1)).sum(1)
        return self.head(z+self.residual(z)).squeeze(-1)

def neural(d,k,rows,query,x,pre,tf,dest):
    m=d['metrics'][k];cls=m in ['AUROC','AUPRC'];yy=torch.tensor(tf.encode(d['y'][rows]),device='cuda')
    xx={n:[torch.tensor(x[n][a],device='cuda') for a in ['fp','dense','aux']] for n in ['fit','query']}
    _,inv,c=np.unique(d['canonical'][d['mi'][rows]],return_inverse=True,return_counts=True)
    weights=torch.tensor(1/np.sqrt(c[inv]),dtype=torch.float32,device='cuda')
    prev=float(yy.mean()) if cls else 1.;pw=min(3.,max(1.,np.sqrt((1-prev)/max(prev,.01))))
    dims=[v.shape[1] for v in xx['fit']];parts=[]
    def predict(net):
        net.eval()
        with torch.inference_mode():return np.concatenate([net(*[a[start:start+512] for a in xx['query']]).cpu().numpy() for start in range(0,len(query),512)])
    for seed in P['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);net=RetrievalNet(dims).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay'])
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,P['epochs'],eta_min=2e-5);snap=[]
        for epoch in range(1,P['epochs']+1):
            net.train();order=rng.permutation(len(rows))
            for start in range(0,len(order),P['batch_size']):
                ix=order[start:start+P['batch_size']];p=net(*[a[ix] for a in xx['fit']]);y=yy[ix];pair=0.
                if cls:
                    loss=nn.functional.binary_cross_entropy_with_logits(p,y,pos_weight=p.new_tensor(pw),reduction='none')
                    pos,neg=p[y>.5],p[y<.5]
                    if len(pos) and len(neg):pair=.08*nn.functional.softplus(neg[None]-pos[:,None]).mean()
                elif m=='MAE':loss=(p-y).abs()
                else:
                    loss=nn.functional.smooth_l1_loss(p,y,beta=.3,reduction='none')
                    diff=y[:,None]-y[None,:];active=diff.abs()>.15
                    if active.any():pair=.15*nn.functional.softplus(-diff[active].sign()*(p[:,None]-p[None,:])[active]).mean()
                value=(loss*weights[ix]).sum()/weights[ix].sum()+pair
                assert torch.isfinite(value);opt.zero_grad(set_to_none=True);value.backward();nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step()
            sched.step()
            if epoch in P['snapshots']:
                p=predict(net);ck=dict(state={n:v.detach().cpu().clone() for n,v in net.state_dict().items()},dimensions=dims,
                    target=tf,preprocessor=pre,signature=signature())
                path=dest/f'neural_{seed}_{epoch}.pt';torch.save(ck,path)
                net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state'])
                np.testing.assert_allclose(p,predict(net),rtol=1e-5,atol=1e-5)
                snap.append(p if m=='Spearman' else tf.decode(p))
        parts.append(np.mean(snap,0));del net
    del xx,yy,weights;torch.cuda.empty_cache();return np.mean(parts,0)

def fit(d,k,rows,query,splits,other,dest):
    dest.mkdir(parents=True,exist_ok=True);sig=signature();path=dest/'predictions.npz'
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],rows);np.testing.assert_array_equal(z['query'],query);return z['p'].copy()
    t0=time.time();x,pre,chosen,tf,audit=features(d,k,rows,query,splits,other)
    pred=[neural(d,k,rows,query,x,pre,tf,dest)]
    m=d['metrics'][k];cls=m in ['AUROC','AUPRC'];klass=CatBoostClassifier if cls else CatBoostRegressor
    yy=tf.encode(d['y'][rows])
    for family in range(2):
        values=[]
        for seed,depth,iters in zip(P['seeds'],P['boost_depths'],P['boost_iterations']):
            params=dict(iterations=iters,depth=depth,learning_rate=P['boost_learning_rate'],l2_leaf_reg=14,
                random_seed=seed,loss_function='Logloss' if cls else ('MAE' if m=='MAE' else 'RMSE'),
                task_type='GPU',devices='0',gpu_ram_part=.2,thread_count=3,border_count=96,
                allow_writing_files=False,verbose=False)
            if cls:params['class_weights']=[1.,min(3.,max(1.,np.sqrt((1-yy.mean())/max(yy.mean(),.01))))]
            model=klass(**params);model.fit(x['fit']['trees'][family],yy)
            f=dest/f'boost_{family}_{seed}.cbm';model.save_model(str(f))
            p=model.predict_proba(x['query']['trees'][family])[:,1] if cls else model.predict(x['query']['trees'][family])
            reload=klass();reload.load_model(str(f));a=reload.predict_proba(x['query']['trees'][family])[:,1] if cls else reload.predict(x['query']['trees'][family])
            np.testing.assert_allclose(p,a,rtol=1e-7,atol=1e-7);values.append(p if cls or m=='Spearman' else tf.decode(p))
        pred.append(np.mean(values,0))
    p=np.stack(pred);assert np.isfinite(p).all()
    np.savez_compressed(path,p=p,fit=rows,query=query,signature=sig)
    audit.update(task=d['tasks'][k],signature=sig,fit_rows=len(rows),query_rows=len(query),seconds=time.time()-t0,
        model_reload_passed=True,test_labels_used=False,local_gpu_training=False)
    b.save(dest/'metadata.json',audit)
    b.emit('v33_fit',task=d['tasks'][k],partition=str(dest.relative_to(OUT)),seconds=round(time.time()-t0));return p

def variants(p,m):
    v=p.copy() if m=='MAE' else np.stack([b.rank(x) for x in p])
    return np.concatenate([p,((v[0]+v[1])*.5)[None],((v[0]+v[2])*.5)[None]],0)

def evaluate(d,other):
    valid,old=baseline(d);f=np.load(b.OUT/'folds.npy');records=[]
    for t in P['tasks']:
        k=d['tasks'].index(t);m=d['metrics'][k]
        tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        vp=variants(fit(d,k,tr,vi,[0],other,OUT/t/'valid'),m)
        screen=max(b.delta(m,d['y'][vi],valid[vi],b.blend(m,valid[vi],p,w)) for p in vp for w in P['weights'])
        record=dict(task=t,metric=m,valid_rows=len(vi),valid_baseline=b.metric(m,d['y'][vi],valid[vi]),
            valid_candidates=[b.metric(m,d['y'][vi],p) for p in vp],screen_gain=screen,selection=None,grid=[])
        if screen>2e-4:
            cp=np.full((5,len(d['y'])),np.nan)
            for fold in range(3):
                q=tr[f[tr]==fold];rows=tr[f[tr]!=fold]
                cp[:,q]=variants(fit(d,k,rows,q,[0],other,OUT/t/f'fold{fold}'),m)
            np.save(OUT/t/'cv_predictions.npy',cp);grid=[]
            for v in range(5):
                for w in P['weights']:
                    gains=[]
                    for fold in range(3):
                        ix=tr[f[tr]==fold];gains.append(b.delta(m,d['y'][ix],old[ix],b.blend(m,old[ix],cp[v,ix],w)))
                    vg=b.delta(m,d['y'][vi],valid[vi],b.blend(m,valid[vi],vp[v],w))
                    objective=float(.55*vg+.45*np.mean(gains)-.04*np.std(gains))
                    grid.append(dict(variant=v,weight=w,cv_gains=gains,cv_mean=float(np.mean(gains)),valid_gain=vg,
                        objective=objective,eligible=bool(vg>2e-4 and np.mean(gains)>0 and objective>5e-4)))
            eligible=[g for g in grid if g['eligible']];record.update(grid=grid,selection=max(eligible,key=lambda g:g['objective']) if eligible else None)
            held=[]
            for fold in range(3):
                choices=[g for g in grid if np.mean([g['cv_gains'][j] for j in range(3) if j!=fold])>5e-4]
                best=max(choices,key=lambda g:np.mean([g['cv_gains'][j] for j in range(3) if j!=fold])) if choices else None
                held.append(best['cv_gains'][fold] if best else 0.)
            record['held_fold_selection_gains']=held
        else:record['rejection']='No positive full-v32 historical-valid blend'
        records.append(record);b.save(OUT/'analysis.json',dict(signature=signature(),status='IN_PROGRESS',tasks=records,limitations=P['limitations']))
        b.emit('v33_selection',task=t,screen_gain=screen,selection=record['selection'])
    b.save(OUT/'analysis.json',dict(signature=signature(),status='EVALUATED',baseline='v32',baseline_score=P['baseline_score'],
        tasks=records,selected_tasks=[r['task'] for r in records if r['selection']],official_score_prediction=None,limitations=P['limitations']))

def final(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    selected=[r for r in report['tasks'] if r['selection']]
    if not selected:return
    other=other_cache(d,[0,1]);compact={}
    for r in selected:
        k=d['tasks'].index(r['task']);rows=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2))
        compact[r['task']]=fit(d,k,rows,q,[0,1],other,OUT/r['task']/'final').tolist()
    b.save(OUT/'test_predictions_compact.json',compact)

def export(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    if not report['selected_tasks']:b.emit('v33_no_candidate');return
    source=next((ROOT/'admet_submissions/v32').glob('*.csv'));assert b.sha(source)==P['baseline_sha256']
    def read(path):return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))
    rows=read(source);template=read(ROOT/'admet_official/submission_template.csv');before=[r.copy() for r in rows]
    compact=json.loads((OUT/'test_predictions_compact.json').read_text());changed={}
    for r in report['tasks']:
        s=r['selection']
        if s is None:continue
        pos=[i for i,x in enumerate(rows) if i>0 and x[0]==r['task']];old=np.array([float(rows[i][4]) for i in pos])
        p=variants(np.array(compact[r['task']]),r['metric'])[s['variant']];assert len(p)==len(old)
        q=b.blend(r['metric'],old,p,s['weight'])
        for i,value in zip(pos,q):rows[i][4]=repr(float(value))
        changed[r['task']]=int(np.sum(q!=old))
    assert len(rows)==len(template)==16380 and rows[0]==template[0]
    for row,t in zip(rows[1:],template[1:]):
        assert len(row)==6 and row[:4]+row[5:]==t[:4]+t[5:] and np.isfinite(float(row[4]))
        if row[5] in ['AUROC','AUPRC']:assert 0<=float(row[4])<=1
    dest=ROOT/'admet_submissions/v33';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261006.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows)
    path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'));assert read(path)==rows
    for x,old in zip(rows[1:],before[1:]):
        if x[0] not in changed:assert x==old
    b.save(OUT/'source_manifest.json',{n:b.sha(ROOT/n) for n in ['admet_v33_run.py','admet_v33_protocol.json','admet_v33_remote_verify.py']})
    b.save(dest/'verification.json',dict(status='EXPERIMENTAL_AWAITING_PLATFORM',signature=signature(),baseline='v32',
        baseline_score=P['baseline_score'],baseline_sha256=b.sha(source),sha256=b.sha(path),rows=16379,columns=6,tasks=22,
        changed_tasks=changed,bytes=path.stat().st_size,encoding='UTF-8 BOM',line_endings='CRLF',template_identity_exact=True,
        training_host=P['training_host'],local_gpu_training=False,platform_score=None,automatically_uploaded=False,limitations=P['limitations']))
    b.emit('v33_export',sha256=b.sha(path),changed_tasks=changed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','evaluate','final','export','all']);args=p.parse_args()
    b.remote_only();assert ROOT==Path(P['remote_root']);OUT.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=True
    b.emit('v33_start',signature=signature(),stage_requested=args.stage);d=b.build_data()
    if args.stage in ['features','evaluate','all']:
        other=other_cache(d,[0])
        if args.stage!='features':evaluate(d,other)
    if args.stage in ['final','all']:final(d)
    if args.stage in ['export','all']:export(d)
    b.save(OUT/f'{args.stage}_completed.json',dict(signature=signature(),completed=True))
