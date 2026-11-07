"""Pharmacophore topology rather than another circular-fragment model.

Fixed two-seed boosted specialists versus identical physicochemical controls.
All features use official structures; no test-label files are opened.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor
from scipy.special import expit
from rdkit import Chem,RDConfig,RDLogger,rdBase
from rdkit.Chem import ChemicalFeatures,rdReducedGraphs
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v108_supervised_transfer as shared
b,r83=shared.data,shared.r83
OUT=ROOT/'admet_model_outputs_v109'
TASKS=('bioavailability_ma','caco2_wang','clearance_hepatocyte_az','clearance_microsome_az',
       'cyp2c9_substrate_carbonmangels','cyp2d6_substrate_carbonmangels','cyp3a4_substrate_carbonmangels',
       'half_life_obach','ppbr_az','vdss_lombardo')
SEEDS=(1094026,1094027);TREES=650;WEIGHT=.35
FACTORY=None

def sig():return hashlib.sha256((b.sha(__file__)+shared.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)

def smiles_list(d):
    smiles=[];seen=set()
    for t in d['tasks']:
        for part in ('train','valid'):
            for r in b.table(ROOT/'admet_competition_data/data_benchmark'/t/(part+'.csv')):
                if r['Drug'] not in seen:seen.add(r['Drug']);smiles.append(r['Drug'])
        for r in d['template']:
            if r['task']==t and r['smiles'] not in seen:seen.add(r['smiles']);smiles.append(r['smiles'])
    assert len(smiles)==len(d['views']['physchem']);return smiles

def molecule_features(sm):
    global FACTORY
    RDLogger.DisableLog('rdApp.*')
    if FACTORY is None:
        FACTORY=SigFactory(ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')),
            minPointCount=2,maxPointCount=2,trianglePruneBins=False)
        FACTORY.SetBins([(0,2),(2,4),(4,6),(6,9),(9,13),(13,20)]);FACTORY.Init()
    import admet_v67_data as parser
    mol,mode=parser.parse_molecule(sm)
    try:
        # Physiochemical role pairs at graph-distance bins; no folded circular IDs.
        p=Generate.Gen2DFingerprint(mol,FACTORY)
        pairs=np.zeros(FACTORY.GetSigSize(),np.float32);pairs[list(p.GetOnBits())]=1.
        erg=rdReducedGraphs.GetErGFingerprint(mol).astype(np.float32)
        return np.r_[pairs,erg],mode,None
    except Exception as ex:
        return np.zeros(216+315,np.float32),mode,type(ex).__name__

def features(d):
    folder=OUT/'features';file=folder/'pharmacophore.npz'
    if file.exists():
        meta=json.loads((folder/'audit.json').read_text());assert meta['signature']==sig() and b.sha(file)==meta['sha256']
        return np.load(file)['x']
    smiles=smiles_list(d);rows=[];errors=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i,(x,mode,error) in enumerate(pool.map(molecule_features,smiles,chunksize=64)):
            assert len(x)==531;rows.append(x)
            if error:errors.append(dict(index=i,parse=mode,error=error))
            if i%5000==0:emit(stage='features',molecules=i,total=len(smiles),seconds=round(time.monotonic()-started))
    x=np.stack(rows);assert len(errors)<.02*len(x),errors[:10]
    folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(file,x=x,signature=sig())
    b.save(folder/'audit.json',dict(status='PASS',signature=sig(),sha256=b.sha(file),shape=list(x.shape),rdkit=rdBase.rdkitVersion,
        smiles_sha256=hashlib.sha256(json.dumps(smiles).encode()).hexdigest(),exceptions=errors,
        source='Official train/valid/template structures; RDKit BaseFeatures rules and reduced graph roles; no learned external weights or data.',test_labels_used=False))
    return x

def load():
    d=shared.load();d['pharmacophore']=features(d);return d

def protocol():
    doc=dict(version=109,signature=sig(),baseline='v101',anchor_sha256=shared.ANCHOR_SHA,tasks=TASKS,seeds=SEEDS,trees=TREES,blend_weight=WEIGHT,
        model='CPU CatBoost depth5,650trees,lr.035,l2=20,random_strength.5,MAE/Logloss/RMSE; fixed seed ensemble.',
        features='216 two-point pharmacophore binary role-pairs +315 reduced pharmacophore graph fuzzy counts +435physicochemical. Fit-only median. No circular fragments or measured assay inputs.',
        control='Same CatBoost parameters and fitting rows using only435physicochemical. One config, no iteration/weight search.',
        selection='Legacy fold0/1. Both blend gains positive and average >1.5%relative MAE/.012rho/.005AUROC/AP; pharmacophore beats physicochemical in both.',
        confirmation='Freeze task list first; fold2/valid positive blend gains; pooled grouped bootstrap95% lower>0; same-row pharmacophore vs control nonnegative on random/scaffold holdouts. No failed fallback.',
        test_labels_used=False,external_data=False,new_pretrained_weights=False,limitations=shared.protocol()['limitations'])
    p=OUT/'protocol.json';OUT.mkdir(exist_ok=True)
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(doc))
    else:b.save(p,doc)

def design(d,k,part,view):
    fit,q=shared.split(d,part);fit=fit[d['tid'][fit]==k];q=q[d['tid'][q]==k]
    raw=d['views']['physchem']
    if view=='pharmacophore':raw=np.c_[raw,d['pharmacophore']]
    xx=raw[d['mi'][fit]].astype(float);qx=raw[d['mi'][q]].astype(float)
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in xx.T])
    x=np.where(np.isfinite(xx),xx,med).astype(np.float32);qx=np.where(np.isfinite(qx),qx,med).astype(np.float32)
    y=d['y'][fit];m=d['metrics'][k]
    if m in ('AUROC','AUPRC'):center,scale=0.,1.;target=y
    elif m=='MAE':
        center=float(np.median(y));scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.001);target=(y-center)/scale
    else:center,scale=0.,1.;target=shared.ndtri(np.clip((shared.rankdata(y)-.5)/len(y),1e-5,1-1e-5))
    return dict(fit=fit,query=q,x=x,qx=qx,median=med,target=target,center=np.array(center),scale=np.array(scale))

def train_one(d,k,part,view,seed):
    folder=OUT/'models'/d['tasks'][k]/part/view/str(seed);file=folder/'predictions.npz';a=design(d,k,part,view)
    if file.exists():
        z=np.load(file);meta=json.loads((folder/'metadata.json').read_text());assert str(z['signature'])==meta['signature']==sig()
        assert b.sha(file)==meta['prediction_sha256'] and b.sha(folder/'model.cbm')==meta['model_sha256']
        np.testing.assert_array_equal(a['fit'],z['fit']);np.testing.assert_array_equal(a['query'],z['query']);return z['raw']
    m=d['metrics'][k];cls=m in ('AUROC','AUPRC');started=time.monotonic()
    net=(CatBoostClassifier if cls else CatBoostRegressor)(iterations=TREES,depth=5,learning_rate=.035,l2_leaf_reg=20,
        random_strength=.5,loss_function='Logloss' if cls else 'MAE' if m=='MAE' else 'RMSE',random_seed=seed,
        thread_count=2,verbose=False,allow_writing_files=False,border_count=64)
    net.fit(a['x'],a['target']);raw=net.predict(a['qx'],prediction_type='RawFormulaVal')
    folder.mkdir(parents=True,exist_ok=True);net.save_model(str(folder/'model.cbm'))
    np.savez_compressed(file,signature=sig(),fit=a['fit'],query=a['query'],raw=raw,median=a['median'],center=a['center'],scale=a['scale'])
    b.save(folder/'metadata.json',dict(signature=sig(),task=d['tasks'][k],part=part,view=view,seed=seed,model_sha256=b.sha(folder/'model.cbm'),prediction_sha256=b.sha(file),seconds=time.monotonic()-started))
    emit(stage='train',task=d['tasks'][k],part=part,view=view,seed=seed,seconds=round(time.monotonic()-started));return raw

def predict(d,k,part,view):
    raw=np.mean([train_one(d,k,part,view,s) for s in SEEDS],axis=0);a=design(d,k,part,view);p=raw*float(a['scale'])+float(a['center'])
    if d['metrics'][k] in ('AUROC','AUPRC'):p=expit(p)
    return p.clip(0,100) if d['tasks'][k]=='ppbr_az' else p

def mix(m,old,p):return (1-WEIGHT)*old+WEIGHT*p if m=='MAE' else (1-WEIGHT)*shared.rankdata(old)/len(old)+WEIGHT*shared.rankdata(p)/len(p)

def develop(d):
    records=[]
    for t in TASKS:
        k=d['tasks'].index(t);m=d['metrics'][k];parts=[]
        for part in ('fold0','fold1'):
            a=design(d,k,part,'pharmacophore');q=a['query'];y=d['y'][q];p=predict(d,k,part,'pharmacophore');c=predict(d,k,part,'control')
            old=shared.baseline(d,k,part);new=mix(m,old,p)
            parts.append(dict(part=part,rows=len(q),baseline_proxy=r83.score(m,y,old),pharmacophore=r83.score(m,y,p),control=r83.score(m,y,c),candidate=r83.score(m,y,new),gain=r83.gain(m,y,old,new),representation_gain=r83.gain(m,y,c,p)))
        selected=bool(all(x['gain']>0 and x['representation_gain']>0 for x in parts) and np.mean([x['gain'] for x in parts])>shared.threshold(m))
        r=dict(task=t,metric=m,parts=parts,selected=selected);records.append(r);emit(stage='development',**r)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records))
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=[r for r in records if r['selected']],confirmation_inspected=False))

def confirm(d):
    f=json.loads((OUT/'frozen_selection.json').read_text());assert f['signature']==sig();records=[]
    for ch in f['selected']:
        k=d['tasks'].index(ch['task']);m=ch['metric'];parts=[];rows=[];yy=[];oo=[];pp=[]
        for part in ('fold0','fold1','fold2','valid','random','scaffold'):
            a=design(d,k,part,'pharmacophore');q=a['query'];y=d['y'][q];p=predict(d,k,part,'pharmacophore');c=predict(d,k,part,'control')
            rec=dict(part=part,rows=len(q),pharmacophore=r83.score(m,y,p),control=r83.score(m,y,c),representation_gain=r83.gain(m,y,c,p))
            if part not in ('random','scaffold'):
                old=shared.baseline(d,k,part);new=mix(m,old,p);rec.update(baseline_proxy=r83.score(m,y,old),candidate=r83.score(m,y,new),gain=r83.gain(m,y,old,new))
                rows.extend(q);yy.extend(y);oo.extend(shared.rankdata(old)/len(old) if m!='MAE' else old);pp.extend(shared.rankdata(new)/len(new) if m!='MAE' else new)
            parts.append(rec)
        ci=shared.bootstrap(d,k,np.array(rows),np.array(yy),np.array(oo),np.array(pp))
        accepted=bool(all(x.get('gain',1)>0 for x in parts) and all(x['representation_gain']>=0 for x in parts if x['part'] in ('random','scaffold')) and ci['percentile_95'][0]>0)
        r=dict(task=ch['task'],metric=m,confirmation=parts,uncertainty=ci,accepted=accepted);records.append(r);emit(stage='confirmation',**r)
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))

def smoke(d):
    ss=smiles_list(d);ids=[0,31,2026,len(ss)-1];x=d['pharmacophore']
    for i in ids:np.testing.assert_array_equal(x[i],molecule_features(ss[i])[0])
    k=d['tasks'].index('caco2_wang');a=design(d,k,'fold0','pharmacophore');saved=d['y'][a['query']].copy();d['y'][a['query']]=1e6
    z=design(d,k,'fold0','pharmacophore');d['y'][a['query']]=saved
    for key in a:np.testing.assert_array_equal(a[key],z[key])
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),query_label_perturbation=True,feature_reconstruction=True,selected_ids=ids))
    emit(stage='smoke',status='PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['protocol','features','smoke','develop','confirm']);args=ap.parse_args();protocol()
    if args.stage!='protocol':
        d=load()
        if args.stage!='features':dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](d)
