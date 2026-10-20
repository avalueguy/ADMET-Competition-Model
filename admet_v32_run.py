"""Remote-only conditional neural and boosted ensembles relative to v29."""
import argparse, csv, hashlib, io, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from catboost import CatBoostClassifier, CatBoostRegressor
import admet_v24_models as b
from admet_v25_run import v24_valid
from admet_v31_run import auxiliary

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_model_outputs_v32'
P=json.loads((ROOT/'admet_v32_protocol.json').read_text(encoding='utf-8'))

def signature():
    names=['admet_v32_run.py','admet_v32_protocol.json','admet_v31_run.py','admet_v31_protocol.json',
           'admet_model_outputs_v29/analysis.json']
    return hashlib.sha256(json.dumps(dict(base=b.signature(),files={n:b.sha(ROOT/n) for n in names}),sort_keys=True).encode()).hexdigest()

def variants(p,m):
    a=p.copy()
    if m!='MAE':a=np.stack([b.rank(v) for v in p])
    return np.concatenate([p, ((a[0]+a[2])*.5)[None], ((a[0]+a[1])*.5)[None]],0)

def previous_predictions(d):
    """Replay the full valid ensemble; CV preserves its newer source weights."""
    report=json.loads((ROOT/'admet_model_outputs_v29/analysis.json').read_text())
    valid=v24_valid(d);cv=np.load(b.OUT/'cv_predictions.npy')[1].copy()
    folds=np.load(b.OUT/'folds.npy')
    gvalid=np.load(ROOT/'admet_model_outputs_v27/valid_predictions.npy')
    gcv=np.load(ROOT/'admet_model_outputs_v27/cv_predictions.npy')
    for record in report['tasks']:
        sel=record['selection']
        if sel is None:continue
        k=d['tasks'].index(record['task']);m=record['metric']
        vi=np.flatnonzero((d['tid']==k)&(d['split']==1));tr=np.flatnonzero((d['tid']==k)&(folds>=0))
        vp=np.zeros(len(vi));cp=np.zeros(len(tr))
        for source in sel['sources']:
            v,j,w=source['version'],source['variant'],source['weight']
            if v==27:x,y=gvalid[j,vi].copy(),gcv[j,tr].copy()
            else:
                path=ROOT/f'admet_model_outputs_v{v}'/record['task']
                zz=np.load(path/'valid/predictions.npz');np.testing.assert_array_equal(zz['query'],vi)
                x=zz['p'][j].copy();y=np.load(path/'cv_predictions.npy')[j,tr].copy()
            if sel['name']=='assay_complement_equal' and m!='MAE':
                x=b.rank(x)
                for fold in range(3):
                    active=folds[tr]==fold;y[active]=b.rank(y[active])
            vp+=w*x;cp+=w*y
        valid[vi]=b.blend(m,valid[vi],vp,sel['weight'])
        for fold in range(3):
            ix=tr[folds[tr]==fold];active=folds[tr]==fold
            cv[ix]=b.blend(m,cv[ix],cp[active],sel['weight'])
        prior=record['historical_valid_baseline']
        actual=b.metric(m,d['y'][vi],valid[vi])
        actual_gain=(prior-actual)/prior if m=='MAE' else actual-prior
        np.testing.assert_allclose(actual_gain,sel['valid_gain'],rtol=1e-8,atol=1e-10)
    np.save(OUT/'baseline_valid.npy',valid);np.save(OUT/'baseline_cv_proxy.npy',cv)
    return valid,cv

class ConditionalNet(nn.Module):
    def __init__(self,fpdim,densedim,assaydim):
        super().__init__();h=P['hidden'];drop=P['dropout']
        def layer(a,z):return nn.Sequential(nn.Linear(a,z),nn.LayerNorm(z),nn.SiLU(),nn.Dropout(drop))
        self.structure=layer(fpdim,h);self.dense=layer(densedim,h)
        self.condition=layer(assaydim,h//2)
        self.gate=nn.Sequential(nn.Linear(2*h+h//2,64),nn.SiLU(),nn.Linear(64,1),nn.Sigmoid())
        self.combine=layer(2*h+h//2,h)
        self.residual=layer(h,h);self.head=nn.Linear(h,1)
    def forward(self,fp,dense,aux):
        n=aux.shape[1]//2;values,mask=aux[:,:n],aux[:,n:]
        if self.training:
            kept=(torch.rand_like(mask)>P['assay_dropout']).to(mask.dtype)
            values,mask=values*kept,mask*kept
        cond=self.condition(torch.cat([values,mask],1))
        a,c=self.structure(fp),self.dense(dense)
        gate=self.gate(torch.cat([a,c,cond],1))*(mask.sum(1,keepdim=True)>0)
        h=self.combine(torch.cat([a,c,cond*gate],1))
        return self.head(h+self.residual(h)).squeeze(-1)

def features(d,k,rows,query,source_splits):
    aux,count,audit=auxiliary(d,source_splits,k)
    raw=np.concatenate([d['desc'],d['emb']],1)
    pre=b.Preprocessor().fit(raw[np.unique(d['mi'][rows])])
    # Materialize only this target's rows, not the full row-expanded matrix.
    normalized=pre.transform(raw)
    dims=d['desc'].shape[1]
    out={}
    fp=np.concatenate([d['fp2'],d['fcfp']],1)
    for name,ix in [('fit',rows),('query',query)]:
        dense=normalized[d['mi'][ix]];assay=aux[ix]
        out[name]=dict(fp=fp[d['mi'][ix]],dense=dense,assay=assay,
                       trees=[np.concatenate([dense,assay],1),np.concatenate([dense[:,:dims],assay],1)])
    return out,pre,count,audit

def neural(d,k,rows,query,x,pre,dest):
    m=d['metrics'][k];cls=m in ['AUROC','AUPRC'];tf=b.Target(d['y'][rows],m)
    y=torch.tensor(tf.encode(d['y'][rows]),device='cuda')
    xx={name:[torch.tensor(x[name][part],device='cuda') for part in ['fp','dense','assay']] for name in ['fit','query']}
    can=d['canonical'][d['mi'][rows]];_,inv,c=np.unique(can,return_inverse=True,return_counts=True)
    ww=torch.tensor(1/np.sqrt(c[inv]),dtype=torch.float32,device='cuda')
    prev=float(y.mean()) if cls else 1.
    pw=min(3.,max(1.,np.sqrt((1-prev)/max(prev,.01))))
    dimensions=[a.shape[1] for a in xx['fit']];parts=[]
    for seed in P['seeds']:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
        net=ConditionalNet(*dimensions).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay'])
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,P['epochs'],eta_min=2e-5);snapshots=[]
        for epoch in range(1,P['epochs']+1):
            net.train();order=rng.permutation(len(rows))
            for start in range(0,len(order),P['batch_size']):
                ix=order[start:start+P['batch_size']];pred=net(*[a[ix] for a in xx['fit']]);true=y[ix];pair=0.
                if cls:
                    losses=nn.functional.binary_cross_entropy_with_logits(pred,true,pos_weight=pred.new_tensor(pw),reduction='none')
                    pos,neg=pred[true>.5],pred[true<.5]
                    if len(pos) and len(neg):pair=.05*nn.functional.softplus(neg[None]-pos[:,None]).mean()
                elif m=='MAE':losses=(pred-true).abs()
                else:
                    losses=nn.functional.smooth_l1_loss(pred,true,beta=.3,reduction='none')
                    diff=true[:,None]-true[None,:];active=diff.abs()>.15
                    if active.any():pair=.1*nn.functional.softplus(-diff[active].sign()*(pred[:,None]-pred[None,:])[active]).mean()
                loss=(losses*ww[ix]).sum()/ww[ix].sum()+pair
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step()
            schedule.step()
            if epoch in P['snapshots']:
                net.eval()
                def predict():
                    with torch.inference_mode():
                        return np.concatenate([net(*[a[start:start+512] for a in xx['query']]).cpu().numpy() for start in range(0,len(query),512)])
                pred=predict();path=dest/f'neural_{seed}_{epoch}.pt'
                torch.save(dict(state={n:v.detach().cpu().clone() for n,v in net.state_dict().items()},
                    target=tf,preprocessor=pre,dimensions=dimensions,signature=signature()),path)
                net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state'])
                np.testing.assert_allclose(pred,predict(),rtol=1e-5,atol=1e-5)
                snapshots.append(pred if m=='Spearman' else tf.decode(pred))
        parts.append(np.mean(snapshots,0));del net
    del xx,y,ww;torch.cuda.empty_cache()
    return np.mean(parts,0)

def fit(d,k,rows,query,source_splits,dest):
    dest.mkdir(parents=True,exist_ok=True);cache=dest/'predictions.npz';sig=signature()
    if cache.exists():
        z=np.load(cache);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],rows);np.testing.assert_array_equal(z['query'],query)
        return z['p'].copy()
    assert not set(rows)&set(query) and np.all(d['split'][rows]<2)
    assert np.all(d['tid'][rows]==k)
    if np.any(d['split'][query]<2):assert not set(d['rg'][rows])&set(d['rg'][query])
    t0=time.time();x,pre,count,audit=features(d,k,rows,query,source_splits)
    pred=[neural(d,k,rows,query,x,pre,dest)]
    m=d['metrics'][k];cls=m in ['AUROC','AUPRC'];tf=b.Target(d['y'][rows],m)
    yy=tf.encode(d['y'][rows]);klass=CatBoostClassifier if cls else CatBoostRegressor
    for family in range(2):
        parts=[]
        for seed,depth in zip(P['seeds'],P['boost_depths']):
            params=dict(iterations=P['boost_iterations'],depth=depth,learning_rate=P['boost_learning_rate'],
                l2_leaf_reg=P['boost_l2'],random_seed=seed,loss_function='Logloss' if cls else ('MAE' if m=='MAE' else 'RMSE'),
                task_type='GPU',devices='0',gpu_ram_part=.2,thread_count=3,border_count=96,verbose=False,allow_writing_files=False)
            if cls:
                prevalence=float(yy.mean());params['class_weights']=[1.,min(3.,max(1.,np.sqrt((1-prevalence)/max(prevalence,.01))))]
            model=klass(**params);model.fit(x['fit']['trees'][family],yy)
            path=dest/f'boost_{family}_{seed}.cbm';model.save_model(str(path))
            q=model.predict_proba(x['query']['trees'][family])[:,1] if cls else model.predict(x['query']['trees'][family])
            restored=klass();restored.load_model(str(path))
            again=restored.predict_proba(x['query']['trees'][family])[:,1] if cls else restored.predict(x['query']['trees'][family])
            np.testing.assert_allclose(q,again,rtol=1e-7,atol=1e-7)
            parts.append(q if cls or m=='Spearman' else tf.decode(q))
        pred.append(np.mean(parts,0))
    p=np.stack(pred);assert np.isfinite(p).all()
    np.savez_compressed(cache,p=p,fit=rows,query=query,signature=sig)
    audit.update(signature=sig,task=d['tasks'][k],fit_rows=len(rows),query_rows=len(query),
        query_auxiliary_coverage=float(np.mean(count[query]>0)),seconds=time.time()-t0,
        model_reload_passed=True,test_labels_used=False,local_gpu_training=False)
    b.save(dest/'metadata.json',audit)
    b.emit('v32_fit',task=d['tasks'][k],partition=str(dest.relative_to(OUT)),seconds=round(time.time()-t0))
    return p

def audit(d):
    records=[]
    for t in ['cyp2c9_substrate_carbonmangels','cyp2c9_veith','ld50_zhu','ppbr_az']:
        k=d['tasks'].index(t);x,count,_=auxiliary(d,[0],k)
        q=dict(d);q['y']=d['y'].copy();q['y'][d['tid']==k]=123456.;q['y'][d['split']==2]=789123.
        again,cc,_=auxiliary(q,[0],k)
        np.testing.assert_array_equal(x,again);np.testing.assert_array_equal(count,cc)
        records.append(dict(task=t,target_and_test_label_perturbation_invariant=True))
    torch.manual_seed(320600)
    net=ConditionalNet(10,12,42).cuda().eval()
    xs=[torch.randn(6,10,device='cuda'),torch.randn(6,12,device='cuda'),torch.zeros(6,42,device='cuda')]
    with torch.inference_mode():
        p=net(*xs).cpu().numpy();s=net(*[x[:1] for x in xs]).cpu().numpy()
        perm=np.array([4,1,0,3,2,5]);pp=net(*[x[perm] for x in xs]).cpu().numpy()
    np.testing.assert_allclose(p[:1],s,rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(p[perm],pp,rtol=1e-5,atol=1e-5)
    b.save(OUT/'isolation_and_batch_audit.json',dict(tasks=records,query_batch_invariance=True,target_mask_removed=True))
    del net;torch.cuda.empty_cache()

def evaluate(d):
    valid,old=previous_predictions(d);folds=np.load(b.OUT/'folds.npy');records=[]
    for t in P['task_priority']:
        k=d['tasks'].index(t);m=d['metrics'][k]
        tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        vp=variants(fit(d,k,tr,vi,[0],OUT/t/'valid'),m)
        screen=max(b.delta(m,d['y'][vi],valid[vi],b.blend(m,valid[vi],q,w)) for q in vp for w in P['weights'])
        record=dict(task=t,metric=m,valid_rows=len(vi),valid_baseline=b.metric(m,d['y'][vi],valid[vi]),
                    valid_candidates=[b.metric(m,d['y'][vi],q) for q in vp],screen_gain=screen,selection=None,grid=[])
        if screen>1e-4:
            cp=np.full((5,len(d['y'])),np.nan)
            for fold in range(3):
                query=tr[folds[tr]==fold];rows=tr[folds[tr]!=fold]
                cp[:,query]=variants(fit(d,k,rows,query,[0],OUT/t/f'fold{fold}'),m)
            np.save(OUT/t/'cv_predictions.npy',cp);grid=[]
            for variant in range(5):
                for weight in P['weights']:
                    gains=[]
                    for fold in range(3):
                        ix=tr[folds[tr]==fold]
                        gains.append(b.delta(m,d['y'][ix],old[ix],b.blend(m,old[ix],cp[variant,ix],weight)))
                    vg=b.delta(m,d['y'][vi],valid[vi],b.blend(m,valid[vi],vp[variant],weight))
                    objective=float(.65*vg+.35*np.mean(gains)-.05*np.std(gains))
                    eligible=vg>1e-4 and np.mean(gains)>0 and sum(x>=0 for x in gains)>=2 and objective>1e-4
                    grid.append(dict(variant=variant,weight=weight,cv_gains=gains,cv_mean=float(np.mean(gains)),
                                     valid_gain=vg,objective=objective,eligible=bool(eligible)))
            options=[g for g in grid if g['eligible']]
            record.update(selection=max(options,key=lambda g:g['objective']) if options else None,grid=grid)
            held=[]
            for fold in range(3):
                others=[j for j in range(3) if j!=fold]
                options=[g for g in grid if np.mean([g['cv_gains'][j] for j in others])>0 and all(g['cv_gains'][j]>=0 for j in others)]
                best=max(options,key=lambda g:np.mean([g['cv_gains'][j] for j in others])) if options else None
                held.append(best['cv_gains'][fold] if best else 0.)
            record['held_fold_cv_selection_gains']=held
        else:record['rejection']='No positive historically reused full-v29 valid blend'
        records.append(record)
        b.save(OUT/'analysis.json',dict(signature=signature(),status='IN_PROGRESS',tasks=records,limitations=P['limitations']))
        b.emit('v32_selection',task=t,screen_gain=screen,selection=record['selection'])
    selected=[r for r in records if r['selection']]
    b.save(OUT/'analysis.json',dict(signature=signature(),status='EVALUATED',tasks=records,baseline='v29',
        baseline_score=P['baseline_score'],selected_tasks=[r['task'] for r in selected],official_score_prediction=None,
        mean_selected_valid_gain=float(np.mean([r['selection']['valid_gain'] for r in selected])) if selected else None,
        mean_selected_cv_proxy_gain=float(np.mean([r['selection']['cv_mean'] for r in selected])) if selected else None,
        limitations=P['limitations']))

def final(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    compact={}
    for record in report['tasks']:
        if record['selection'] is None:continue
        k=d['tasks'].index(record['task'])
        rows=np.flatnonzero((d['tid']==k)&(d['split']<2));query=np.flatnonzero((d['tid']==k)&(d['split']==2))
        compact[record['task']]=fit(d,k,rows,query,[0,1],OUT/record['task']/'final').tolist()
    b.save(OUT/'test_predictions_compact.json',compact)

def export(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature()
    if not report['selected_tasks']:b.emit('v32_no_candidate');return
    source=next((ROOT/'admet_submissions/v29').glob('*.csv'));assert b.sha(source)==P['baseline_sha256']
    def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
    rows=read(source);before=[r.copy() for r in rows];template=read(ROOT/'admet_official/submission_template.csv')
    predictions=json.loads((OUT/'test_predictions_compact.json').read_text());changed={}
    for record in report['tasks']:
        sel=record['selection']
        if sel is None:continue
        t=record['task'];pos=[i for i,r in enumerate(rows) if i>0 and r[0]==t]
        base=np.array([float(rows[i][4]) for i in pos]);p=variants(np.array(predictions[t]),record['metric'])[sel['variant']]
        assert len(p)==len(base)
        q=b.blend(record['metric'],base,p,sel['weight'])
        for i,v in zip(pos,q):rows[i][4]=repr(float(v))
        changed[t]=int(np.sum(q!=base))
    assert len(rows)==len(template)==16380 and rows[0]==template[0]
    for r,t in zip(rows[1:],template[1:]):
        assert len(r)==6 and r[:4]+r[5:]==t[:4]+t[5:] and np.isfinite(float(r[4]))
        if r[5] in ['AUROC','AUPRC']:assert 0<=float(r[4])<=1
    dest=ROOT/'admet_submissions/v32';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261006-v32.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows)
    path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'));actual=read(path)
    assert actual==rows
    for a,p in zip(actual[1:],before[1:]):
        if a[0] not in changed:assert a==p
    manifest={n:b.sha(ROOT/n) for n in ['admet_v32_run.py','admet_v32_protocol.json','admet_v31_run.py','admet_v31_protocol.json']}
    b.save(OUT/'source_manifest.json',manifest)
    b.save(dest/'verification.json',dict(status='EXPERIMENTAL_AWAITING_PLATFORM',signature=signature(),baseline='v29',
        baseline_score=P['baseline_score'],baseline_sha256=b.sha(source),sha256=b.sha(path),rows=16379,tasks=22,columns=6,
        changed_tasks=changed,bytes=path.stat().st_size,encoding='UTF-8 BOM',line_endings='CRLF',template_identity_exact=True,
        training_host=P['training_host'],local_gpu_training=False,platform_score=None,automatically_uploaded=False,
        limitations=P['limitations']))
    b.emit('v32_export',sha256=b.sha(path),changed_tasks=changed)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','evaluate','final','export','all']);args=parser.parse_args()
    b.remote_only();assert ROOT==Path(P['remote_root']);OUT.mkdir(parents=True,exist_ok=True)
    b.emit('v32_start',signature=signature(),stage_requested=args.stage);d=b.build_data()
    if args.stage in ['audit','all']:audit(d)
    if args.stage in ['evaluate','all']:evaluate(d)
    if args.stage in ['final','all']:final(d)
    if args.stage in ['export','all']:export(d)
    b.save(OUT/f'{args.stage}_completed.json',dict(signature=signature(),completed=True))
