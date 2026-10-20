"""Remote property-prior residual experts, ring specialization, honest source controls."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS', '3')
import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import torch
from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski
import admet_v24_models as b
import admet_v44_run as pair

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'admet_model_outputs_v51'
P = json.loads((ROOT/'admet_v51_protocol.json').read_text(encoding='utf-8'))


def signature():
    return hashlib.sha256(json.dumps(dict(base=b.signature(),
        files={n:b.sha(ROOT/n) for n in ['admet_v51_run.py','admet_v51_protocol.json']}),sort_keys=True).encode()).hexdigest()


def chemistry(d):
    path=OUT/'physchem.npz'
    if path.exists():
        z=np.load(path)
        assert str(z['signature'])==signature()
        return {n:z[n].copy() for n in ['prior','cyclic','fps','dense']}
    priors=[]; cyclic=[]
    for s in d['smiles']:
        m=Chem.MolFromSmiles(s)
        if m is None:
            priors.append([0.,0.]);cyclic.append(False);continue
        lp=Crippen.MolLogP(m);mw=Descriptors.MolWt(m)
        rb=Lipinski.NumRotatableBonds(m)
        ap=sum(a.GetIsAromatic() for a in m.GetAtoms())/max(1,m.GetNumHeavyAtoms())
        esol=.16-.63*lp-.0062*mw+.066*rb-.74*ap
        priors.append([lp,esol]);cyclic.append(m.GetRingInfo().NumRings()>0)
    extended=np.load(ROOT/'admet_model_outputs_v37/features/chemistry.npz')['x']
    parent=np.load(ROOT/'admet_model_outputs_v41/features/parents.npz')
    fps=np.concatenate([d['fp2'],d['fcfp'],extended,parent['fp']],1)
    dense=np.concatenate([d['desc'],d['emb'],parent['desc'],parent['flags'],np.asarray(priors)],1).astype(np.float32)
    x=dict(prior=np.asarray(priors,np.float32),cyclic=np.asarray(cyclic),fps=fps,dense=dense)
    np.savez_compressed(path,signature=signature(),**x)
    return x


def baseline(d):
    # These saved arrays already replay v44 and local v45 correctly. Only apply v46 selections.
    valid=np.load(ROOT/'admet_model_outputs_v46/baseline_valid.npy').copy()
    cv=np.load(ROOT/'admet_model_outputs_v46/baseline_cv_proxy.npy').copy()
    report=json.loads((ROOT/'admet_model_outputs_v46/analysis.json').read_text())
    folds=np.load(b.OUT/'folds.npy')
    for r in report['tasks']:
        s=r.get('selection')
        if not s:continue
        k=d['tasks'].index(r['task']);m=r['metric']
        vi=np.flatnonzero((d['tid']==k)&(d['split']==1));tr=np.flatnonzero((d['tid']==k)&(d['split']==0))
        vp=np.load(ROOT/'admet_model_outputs_v46'/r['task']/'valid_predictions.npy')[s['variant']]
        cp=np.load(ROOT/'admet_model_outputs_v46'/r['task']/'cv_predictions.npy')[s['variant']]
        valid[vi]=b.blend(m,valid[vi],vp,s['weight'])
        for f in range(3):
            ix=tr[folds[tr]==f];cv[ix]=b.blend(m,cv[ix],cp[ix],s['weight'])
    source=next((ROOT/'admet_submissions/v46').glob('*.csv'))
    # Score linkage is unconfirmed; source identity is explicit and never inferred from score.
    assert b.sha(source)==P['baseline_sha256'], (b.sha(source),'baseline mismatch')
    b.save(OUT/'baseline.json',dict(csv=str(source),sha256=b.sha(source),official_score=None,
        current_score_unknown=True,valid_cv='v46 exactly replayed; historical ensemble CV proxy'))
    return valid,cv


def design(d,c,k,rows):
    mol=np.unique(d['mi'][rows])
    # Quotas keep path/torsion views even when high-variance Morgan columns dominate.
    boundaries=[0,2048,4096,8192,10240,11264,12288,14336,16384]
    chosen=[]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        var=np.var(c['fps'][mol,lo:hi],0)
        chosen.extend(lo+np.argsort(var,kind='stable')[-P['fingerprint_columns']//8:])
    chosen=np.asarray(chosen,int)
    pre=b.Preprocessor().fit(c['dense'][mol])
    x=np.concatenate([pre.transform(c['dense']),c['fps'][:,chosen]],1)
    name=d['tasks'][k]
    prior=c['prior'][:,0] if name=='lipophilicity_astrazeneca' else (c['prior'][:,1] if name=='solubility_aqsoldb' else np.zeros(len(x),np.float32))
    return x,prior,pre,chosen


def fit(d,c,k,rows,query,dest,families=None):
    dest.mkdir(parents=True,exist_ok=True);path=dest/'predictions.npz';sig=signature()
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig
        np.testing.assert_array_equal(rows,z['fit']);np.testing.assert_array_equal(query,z['query'])
        return z['p'].copy()
    assert not set(rows)&set(query) and np.all(d['split'][rows]<2)
    if np.any(d['split'][query]<2):
        assert not set(d['rg'][rows])&set(d['rg'][query])
        assert not set(d['canonical'][d['mi'][rows]])&set(d['canonical'][d['mi'][query]])
    x,prior,pre,chosen=design(d,c,k,rows)
    start=time.time();families=list(range(3)) if families is None else families
    pred=np.full((3,len(query)),np.nan)
    for family in families:
        # For endpoints without a physical prior, the second family is a complementary seed/depth ensemble.
        use=rows[c['cyclic'][d['mi'][rows]]] if family==2 else rows
        if len(use)<80:use=rows
        ids=d['mi'][use]
        _,inv,count=np.unique(d['canonical'][ids],return_inverse=True,return_counts=True)
        weight=1./count[inv]
        offset=prior if family>0 else np.zeros_like(prior)
        y=d['y'][use]-offset[ids]
        values=[]
        seeds=zip(P['seeds'],P['depths'],P['iterations'])
        for seed,depth,iters in seeds:
            model=CatBoostRegressor(iterations=iters,depth=depth,learning_rate=P['learning_rate'],
                loss_function='MAE',eval_metric='MAE',l2_leaf_reg=8,random_seed=seed+100*family,
                task_type='GPU',devices='0',gpu_ram_part=.22,thread_count=3,border_count=128,
                verbose=False,allow_writing_files=False)
            model.fit(x[ids],y,sample_weight=weight)
            file=dest/f'expert{family}_{seed}.cbm';model.save_model(str(file))
            v=model.predict(x[d['mi'][query]])+offset[d['mi'][query]]
            restored=CatBoostRegressor();restored.load_model(str(file))
            np.testing.assert_allclose(v,restored.predict(x[d['mi'][query]])+offset[d['mi'][query]],atol=1e-8,rtol=1e-8)
            values.append(v)
            del model,restored
        pred[family]=np.mean(values,0)
        b.emit('v51_fit',task=d['tasks'][k],partition=dest.name,family=family,fit_rows=len(use),seconds=round(time.time()-start))
    np.savez_compressed(path,p=pred,fit=rows,query=query,signature=sig)
    torch.save(dict(preprocessor=pre,chosen=chosen,families=families,signature=sig),dest/'preprocessing.pt')
    b.save(dest/'metadata.json',dict(signature=sig,fit_rows=len(rows),query_rows=len(query),
        families=families,scaffold_overlap=0,model_reload_passed=True,preprocessor_fit_only=True,
        no_auxiliary_labels=True,test_labels_used=False,local_gpu_training=False,seconds=time.time()-start))
    return pred


def variants(p):
    return np.concatenate([p,((p[0]+p[1])/2)[None],((p[1]+p[2])/2)[None]])


def evaluate(d,c,tasks):
    valid,cv=baseline(d);folds=np.load(b.OUT/'folds.npy')
    records=[]
    for task in tasks:
        k=d['tasks'].index(task);tr=np.flatnonzero((d['tid']==k)&(d['split']==0))
        vi=np.flatnonzero((d['tid']==k)&(d['split']==1));te=np.flatnonzero((d['tid']==k)&(d['split']==2))
        cp=np.full((5,len(d['y'])),np.nan)
        for f in range(3):
            q=tr[folds[tr]==f];rr=tr[folds[tr]!=f]
            cp[:,q]=variants(fit(d,c,k,rr,q,OUT/task/f'fold{f}'))
        np.save(OUT/task/'cv_predictions.npy',cp)
        vp=variants(fit(d,c,k,tr,vi,OUT/task/'valid'))
        ring_v=float(c['cyclic'][d['mi'][vi]].mean());ring_t=float(c['cyclic'][d['mi'][te]].mean())
        shifted=abs(ring_v-ring_t)>.5
        grid=[]
        for v in range(5):
            for w in P['weights']:
                gains=[b.delta('MAE',d['y'][ix],cv[ix],b.blend('MAE',cv[ix],cp[v,ix],w))
                    for f in range(3) for ix in [tr[folds[tr]==f]]]
                vg=b.delta('MAE',d['y'][vi],valid[vi],b.blend('MAE',valid[vi],vp[v],w))
                objective=float(np.mean(gains)-.12*np.std(gains)) if shifted else float(.85*np.mean(gains)+.15*vg-.12*np.std(gains))
                grid.append(dict(variant=v,weight=w,cv_gains=gains,cv_mean=float(np.mean(gains)),valid_gain=vg,objective=objective))
        held=[]
        for f in range(3):
            best=max(grid,key=lambda x:np.mean([g for j,g in enumerate(x['cv_gains']) if j!=f]))
            held.append(best['cv_gains'][f])
        good=[x for x in grid if x['cv_mean']>.001 and x['objective']>.001 and sum(g>0 for g in x['cv_gains'])>=2]
        selected=max(good,key=lambda x:x['objective']) if good and np.mean(held)>.001 else None
        r=dict(task=task,metric='MAE',selection=selected,grid=grid,held_fold_selection_gains=held,
            held_fold_mean=float(np.mean(held)),valid_cyclic=ring_v,test_cyclic=ring_t,domain_shift=shifted)
        records.append(r)
        b.save(OUT/'analysis.json',dict(signature=signature(),status='IN_PROGRESS',tasks=records,limitations=P['limitations']))
        b.emit('v51_selection',task=task,selection=selected,held_fold_mean=r['held_fold_mean'])
    b.save(OUT/'analysis.json',dict(signature=signature(),status='FROZEN',tasks=records,limitations=P['limitations']))


def fresh(d,c):
    report=json.loads((OUT/'analysis.json').read_text());assert report['status']=='FROZEN'
    records=[]
    old_report=json.loads((pair.OUT/'analysis.json').read_text())
    for r in report['tasks']:
        s=r['selection']
        if not s:continue
        task=r['task'];k=d['tasks'].index(task)
        allrows=np.flatnonzero((d['tid']==k)&(d['split']<2))
        # Test-like cyclic holdout chosen without labels. All acyclic rows remain eligible fit rows.
        groups=np.unique(d['rg'][allrows][c['cyclic'][d['mi'][allrows]]])
        rng=np.random.default_rng(P['fresh_seed']+k)
        held=rng.permutation(groups)[:max(1,int(.25*len(groups)))]
        active=np.isin(d['rg'][allrows],held)
        rr,q=allrows[~active],allrows[active]
        pred=variants(fit(d,c,k,rr,q,OUT/task/'fresh'))[s['variant']]
        # Retrain the actual v44 source architecture under this new partition, excluding held labels.
        controls=pair.variants(pair.fit(d,k,rr,q,pair.OUT/task/'v51_fresh_control'),'MAE')
        old=next(x for x in old_report['tasks'] if x['task']==task)
        ctrl=controls[old['selection']['variant']] if old['selection'] else controls[0]
        blend=b.blend('MAE',ctrl,pred,s['weight'])
        gain=b.delta('MAE',d['y'][q],ctrl,blend)
        records.append(dict(task=task,fit_rows=len(rr),query_rows=len(q),
            control_mae=b.metric('MAE',d['y'][q],ctrl),candidate_mae=b.metric('MAE',d['y'][q],pred),
            blend_mae=b.metric('MAE',d['y'][q],blend),blend_gain=gain,pass_gate=bool(gain>0),
            control='v44 selected source freshly retrained; not full ensemble',scaffold_overlap=0))
        b.save(OUT/'fresh.json',dict(signature=signature(),status='IN_PROGRESS',tasks=records))
        b.emit('v51_fresh',**records[-1])
    b.save(OUT/'fresh.json',dict(signature=signature(),status='COMPLETE',tasks=records))


def export(d,c):
    report=json.loads((OUT/'analysis.json').read_text());fresh_report=json.loads((OUT/'fresh.json').read_text())
    assert report['status']=='FROZEN' and fresh_report['status']=='COMPLETE'
    assert report['signature']==fresh_report['signature']==signature()
    passed={r['task'] for r in fresh_report['tasks'] if r['pass_gate']}
    source=next((ROOT/'admet_submissions/v46').glob('*.csv'));assert b.sha(source)==P['baseline_sha256']
    def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
    rows=read(source);template=read(ROOT/'admet_official/submission_template.csv')
    replay={};changed={}
    for r in report['tasks']:
        s=r['selection'];task=r['task']
        if not s or task not in passed:continue
        k=d['tasks'].index(task);rr=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2))
        p=fit(d,c,k,rr,q,OUT/task/'final');branch=variants(p)[s['variant']]
        pos=[i for i,row in enumerate(rows) if i and row[0]==task]
        np.testing.assert_array_equal([int(rows[i][1]) for i in pos],d['row'][q])
        old=np.asarray([float(rows[i][4]) for i in pos]);new=b.blend('MAE',old,branch,s['weight'])
        for i,v in zip(pos,new):rows[i][4]=repr(float(v))
        changed[task]=int(np.sum(old!=new));replay[task]=dict(selection=s,old=old.tolist(),source=p.tolist(),result=new.tolist())
    if not changed:
        b.save(OUT/'no_release.json',dict(status='NO_VERIFIED_IMPROVEMENT',signature=signature()));return
    assert len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,z in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==z[:4]+z[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v51';dest.mkdir(parents=True,exist_ok=True)
    file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261016-v51.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows)
    raw=b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8')
    if file.exists():assert file.read_bytes()==raw,'Never overwrite a different release'
    else:file.write_bytes(raw)
    b.save(OUT/'fusion_replay.json',replay)
    b.save(dest/'verification.json',dict(status='EXPERIMENTAL_AWAITING_PLATFORM',signature=signature(),
        sha256=b.sha(file),baseline_sha256=b.sha(source),baseline='v46',current_official_score=None,
        changed_tasks=changed,rows=16379,columns=6,tasks=22,bytes=len(raw),encoding='UTF-8 BOM',line_endings='CRLF',
        model_training_completed=True,local_gpu_training=False,automatically_uploaded=False,limitations=P['limitations']))
    b.emit('v51_export',path=str(file),sha256=b.sha(file),changed_tasks=changed)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','evaluate','fresh','export','smoke']);p.add_argument('--tasks',nargs='+')
    a=p.parse_args();b.remote_only();assert ROOT==Path(P['remote_root'])
    OUT.mkdir(parents=True,exist_ok=True);torch.backends.cuda.matmul.allow_tf32=False
    d=b.build_data();c=chemistry(d);tasks=a.tasks or P['tasks']
    if a.stage=='smoke':
        k=d['tasks'].index(tasks[0]);f=np.load(b.OUT/'folds.npy');tr=np.flatnonzero((d['tid']==k)&(d['split']==0))
        fit(d,c,k,tr[f[tr]!=0],tr[f[tr]==0],OUT/tasks[0]/'fold0');return
    if a.stage in ['all','evaluate']:evaluate(d,c,tasks)
    if a.stage in ['all','fresh']:fresh(d,c)
    if a.stage in ['all','export']:export(d,c)
    b.save(OUT/(a.stage+'_completed.json'),dict(signature=signature(),completed=True))


if __name__=='__main__':main()
