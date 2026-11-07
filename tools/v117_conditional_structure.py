"""Official measured-assay context with full molecular features.

Targets and target masks are absent from every auxiliary feature. Same-row
descriptor+assay controls measure whether full structural fragments help.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='3'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import xgboost as xgb
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v115_diverse_specialists as prev
import v96_normalized_assays as assays
b,r83=prev.b,prev.r83
OUT=ROOT/'admet_model_outputs_v117'
TASKS=('caco2_wang','clearance_hepatocyte_az','clearance_microsome_az',
    'half_life_obach','vdss_lombardo','herg','bbb_martins',
    'cyp3a4_substrate_carbonmangels','solubility_aqsoldb')
SEEDS=(1174026,1174027);DEPTHS=(3,6);WEIGHT=.5;TREES=800

def sig():return hashlib.sha256((b.sha(__file__)+prev.sig()+assays.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def save(name,x):b.save(OUT/name,x)
def protocol():
    doc=dict(version=117,signature=sig(),tasks=TASKS,seeds=SEEDS,depths=DEPTHS,trees=TREES,weight=WEIGHT,
        features='All12288 logcount fingerprint and435physchem, target-excluded exact/neutral other-assay values,masks and spreads. No historical predictions as model inputs.',
        model='CPU XGBoost histogram, depth3/6, eta.025, lambda30, alpha.1, minchild6, maxbin64, .85 rows, .8 cols, MAE/log1pRMSE/logloss.',
        control='Same fit rows and parameters using435physchem+same assay columns. OOF used only to compare the fixed final blend.',
        selection='Freeze depth onfold0/1 with positive blend and structure-control gains; maxminimum blend gain above1.2%MAE/.008rho/.0025AUC.',
        confirmation='All4 blend gains positive and mean above threshold, pooled same-row control gain positive. No retuning after failure.',
        test_labels_used=False,external_data=False,conditional_official_measurements=True,
        limitations='Public labels historically reused; historical proxy not complete v114 OOF. Measured other assays can exist on query molecules. No official score estimate.')
    OUT.mkdir(exist_ok=True);p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(doc))
    else:b.save(p,doc)

def design(d,k,part,view):
    fit,q=r83.split(d,k,part);aux,cols,source,counts,exact=assays.assays(d,k,part)
    assert k not in cols and np.all(d['tid'][source]!=k)
    raw=d['views']['multiview' if view=='full' else 'physchem']
    x=np.c_[aux[fit],raw[d['mi'][fit]]].astype(np.float32)
    qx=np.c_[aux[q],raw[d['mi'][q]]].astype(np.float32)
    x[~np.isfinite(x)]=np.nan;qx[~np.isfinite(qx)]=np.nan
    return fit,q,x,qx,source

def fit(d,k,part,depth,view):
    folder=OUT/'models'/d['tasks'][k]/part/f'{view}_depth{depth}';file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],r83.split(d,k,part)[1]);return z['p'].copy()
    rows,q,x,qx,source=design(d,k,part,view);y=d['y'][rows];m=d['metrics'][k]
    target=np.log1p(y) if m=='Spearman' else y
    obj='binary:logistic' if m in ('AUROC','AUPRC') else 'reg:absoluteerror' if m=='MAE' else 'reg:squarederror'
    folder.mkdir(parents=True,exist_ok=True);members=[];started=time.monotonic()
    for seed in SEEDS:
        model=xgb.train(dict(objective=obj,max_depth=depth,eta=.025,reg_lambda=30.,reg_alpha=.1,
            min_child_weight=6.,subsample=.85,colsample_bytree=.8,max_bin=64,nthread=3,
            seed=seed,device='cpu',tree_method='hist',verbosity=0),xgb.DMatrix(x,label=target),num_boost_round=TREES)
        p=model.predict(xgb.DMatrix(qx));mp=folder/f'model_{seed}.ubj';model.save_model(mp)
        replay=xgb.Booster();replay.load_model(mp);rp=replay.predict(xgb.DMatrix(qx));np.testing.assert_array_equal(p,rp);members.append(p)
    p=np.mean(members,axis=0).astype(float)
    np.savez_compressed(file,signature=sig(),fit=rows,query=q,source=source,p=p)
    emit(stage='train',task=d['tasks'][k],part=part,depth=depth,view=view,seconds=round(time.monotonic()-started))
    return p

def record(d,k,part,depth):
    m=d['metrics'][k];q=r83.split(d,k,part)[1];y=d['y'][q];a=prev.base(d,k,part)
    z=fit(d,k,part,depth,'full');c=fit(d,k,part,depth,'control')
    if m=='MAE':p=(1-WEIGHT)*a+WEIGHT*z
    else:p=(1-WEIGHT)*r83.rankdata(a)/len(a)+WEIGHT*r83.rankdata(z)/len(z)
    return dict(part=part,rows=len(q),baseline_proxy=r83.score(m,y,a),expert=r83.score(m,y,z),
        descriptor_assay_control=r83.score(m,y,c),candidate=r83.score(m,y,p),gain=r83.gain(m,y,a,p),structure_gain=r83.gain(m,y,c,z))

def develop(d):
    protocol();records=[];selected=[]
    for task in TASKS:
        k=d['tasks'].index(task);m=d['metrics'][k];choices=[]
        for depth in DEPTHS:
            parts=[record(d,k,p,depth) for p in ('fold0','fold1')]
            choices.append(dict(task=task,metric=m,depth=depth,parts=parts,
                objective=min(p['gain'] for p in parts),structure_positive=all(p['structure_gain']>0 for p in parts)))
        valid=[r for r in choices if r['structure_positive']]
        best=max(valid or choices,key=lambda r:r['objective'])
        if best['objective']>prev.threshold(m) and best['structure_positive']:selected.append(best)
        records.extend(choices);emit(stage='development',**best)
        save('development.json',dict(status='IN_PROGRESS',records=records))
    save('development.json',dict(status='COMPLETE',signature=sig(),records=records))
    save('frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected))

def confirm(d):
    f=json.loads((OUT/'frozen_selection.json').read_text());assert f['signature']==sig();records=[]
    for r in f['selected']:
        k=d['tasks'].index(r['task']);parts=[record(d,k,p,r['depth']) for p in ('fold0','fold1','fold2','valid')]
        rr=dict(**r,confirmation=parts,accepted=bool(all(p['gain']>0 for p in parts) and np.mean([p['gain'] for p in parts])>prev.threshold(r['metric']) and np.mean([p['structure_gain'] for p in parts])>0))
        records.append(rr);emit(stage='confirmation',**rr)
        save('confirmation.json',dict(status='IN_PROGRESS',records=records))
    save('confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']]))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','confirm']);args=ap.parse_args()
    dict(develop=develop,confirm=confirm)[args.stage](prev.load())
