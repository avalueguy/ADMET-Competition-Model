"""Frozen scaffold selection and final multi-task GINE release, remote only."""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[_k]='2'
import argparse,csv,io,json,time
from pathlib import Path
import numpy as np,torch
import admet_v24_models as b
import admet_v27_models as graph
import admet_v61_run as v61
import admet_v60_baseline as anchor
from admet_v62_model import ROOT,OUT,P,signature,fit,selftest

def variants(p,m):
    q=p.copy() if m=='MAE' else np.stack([b.rank(x) for x in p]);return np.concatenate([p,np.mean(q,0)[None]])

def evaluate(d,g):
    f=np.load(b.OUT/'folds.npy');cv=np.full((2,len(d['y'])),np.nan);vp=np.full_like(cv,np.nan)
    for fold in range(3):
        q=np.flatnonzero(f==fold);rr=np.flatnonzero((f>=0)&(f!=fold));cv[:,q]=np.mean([fit(d,g,rr,q,seed,OUT/'cv'/f'fold{fold}'/str(seed)) for seed in P['seeds']],0);np.save(OUT/'cv_predictions.npy',cv)
    q=np.flatnonzero(d['split']==1);rr=np.flatnonzero((d['split']==0)&~np.isin(d['rg'],np.unique(d['rg'][q])))
    vp[:,q]=np.mean([fit(d,g,rr,q,seed,OUT/'valid'/str(seed)) for seed in P['seeds']],0);np.save(OUT/'valid_predictions.npy',vp)
    bv,bc=v61.baseline(d);np.save(OUT/'baseline_valid.npy',bv);np.save(OUT/'baseline_cv_proxy.npy',bc);records=[]
    for k,task in enumerate(d['tasks']):
        m=d['metrics'][k];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1));cp=np.full((3,len(d['y'])),np.nan)
        for fold in range(3):
            ix=tr[f[tr]==fold];cp[:,ix]=variants(cv[:,ix],m)
        vv=variants(vp[:,vi],m);grid=[]
        for j in range(3):
            for w in P['weights']:
                gains=[b.delta(m,d['y'][ix],bc[ix],b.blend(m,bc[ix],cp[j,ix],w)) for fold in range(3) for ix in [tr[f[tr]==fold]]]
                grid.append(dict(variant=j,weight=w,cv_gains=gains,cv_mean=float(np.mean(gains)),objective=float(np.mean(gains)-.1*np.std(gains)),valid_gain=b.delta(m,d['y'][vi],bv[vi],b.blend(m,bv[vi],vv[j],w))))
        held=[]
        for fold in range(3):
            eligible=[r for r in grid if np.mean([x for j,x in enumerate(r['cv_gains']) if j!=fold])>.002]
            s=max(eligible,key=lambda r:np.mean([x for j,x in enumerate(r['cv_gains']) if j!=fold])) if eligible else None;held.append(s['cv_gains'][fold] if s else 0.)
        eligible=[r for r in grid if r['cv_mean']>.002 and r['objective']>.002 and sum(x>0 for x in r['cv_gains'])>=2]
        s=max(eligible,key=lambda r:r['objective']) if eligible and np.mean(held)>.002 else None
        records.append(dict(task=task,metric=m,selection=s,held_gains=held,held_mean=float(np.mean(held)),grid=grid));b.emit('v62_selection',task=task,selection=s,held_mean=float(np.mean(held)))
    b.save(OUT/'analysis.json',dict(status='FROZEN',signature=signature(),tasks=records,limitations=P['limitations']));return records

def fresh(d,g,records):
    selected=[r for r in records if r['selection']]
    if not selected:b.save(OUT/'fresh.json',dict(status='COMPLETE',signature=signature(),tasks=[]));return []
    allrows=np.flatnonzero(d['split']<2);groups=np.unique(d['rg'][allrows]);held=np.random.default_rng(P['fresh_seed']).permutation(groups)[:max(1,int(.25*len(groups)))];mask=np.isin(d['rg'][allrows],held);rr,q=allrows[~mask],allrows[mask]
    pp=fit(d,g,rr,q,P['seeds'][0],OUT/'fresh'/str(P['seeds'][0]))
    control=graph.fit(d,g,rr,q,270601,graph.OUT/'v62_fresh_control'/'270601');results=[]
    for r in selected:
        k=d['tasks'].index(r['task']);m=r['metric'];s=r['selection'];active=d['tid'][q]==k;ix=q[active];p=variants(pp[:,active],m)[s['variant']];ctrl=control[1,active];merged=b.blend(m,ctrl,p,s['weight']);gain=b.delta(m,d['y'][ix],ctrl,merged)
        results.append(dict(**r,fresh_gain=gain,fresh_control=b.metric(m,d['y'][ix],ctrl),fresh_blend=b.metric(m,d['y'][ix],merged),fresh_rows=len(ix),pass_gate=bool(gain>.001),control_is_full_ensemble=False));b.emit('v62_fresh',task=r['task'],gain=gain,rows=len(ix),pass_gate=bool(gain>.001))
    b.save(OUT/'fresh.json',dict(status='COMPLETE',signature=signature(),tasks=results,scaffold_overlap=0,control='Freshly retrained v27 one seed with identical fit/query rows',limitations=P['limitations']));return results

def export(d,g,records):
    passed=[r for r in records if r['pass_gate']]
    if not passed:b.save(OUT/'no_release.json',dict(status='NO_VERIFIED_GAIN',signature=signature()));return
    source=next((ROOT/'admet_submissions/v60').glob('*.csv'));assert b.sha(source)==P['baseline_sha256'];rows=anchor.rows(source);template=anchor.rows(ROOT/'admet_official/submission_template.csv');rr=np.flatnonzero(d['split']<2);q=np.flatnonzero(d['split']==2)
    pp=np.mean([fit(d,g,rr,q,seed,OUT/'final'/str(seed)) for seed in P['seeds']],0);np.save(OUT/'test_predictions.npy',pp);replay={};changed={}
    for r in passed:
        task=r['task'];k=d['tasks'].index(task);m=r['metric'];s=r['selection'];active=d['tid'][q]==k;p=variants(pp[:,active],m)[s['variant']];pos=[i for i,a in enumerate(rows) if i and a[0]==task];np.testing.assert_array_equal([int(rows[i][1]) for i in pos],d['row'][q[active]]);old=np.array([float(rows[i][4]) for i in pos]);new=b.blend(m,old,p,s['weight'])
        if task=='ppbr_az':new=new.clip(0,100)
        for i,value in zip(pos,new):rows[i][4]=repr(float(value))
        changed[task]=int(np.sum(new!=old));replay[task]=dict(metric=m,selection=s,baseline=old.tolist(),branch=p.tolist(),result=new.tolist(),evidence=r)
    assert len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,t in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==t[:4]+t[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v62';dest.mkdir(parents=True,exist_ok=True);file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261017.csv';buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows);raw=b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8')
    if file.exists():assert file.read_bytes()==raw
    else:file.write_bytes(raw)
    b.save(OUT/'fusion_replay.json',replay);b.save(OUT/'release.json',dict(status='AWAITING_PLATFORM',signature=signature(),selected={r['task']:r for r in passed},sha256=b.sha(file),official_score=None,limitations=P['limitations']));b.save(dest/'verification.json',dict(status='AWAITING_PLATFORM_EVALUATION',signature=signature(),sha256=b.sha(file),baseline_sha256=b.sha(source),rows=16379,columns=6,tasks=22,changed_tasks=changed,bytes=len(raw),encoding='UTF-8 BOM',line_endings='CRLF',official_score=None,known_score_upper_bound=50.804792,local_gpu_training=False,automatically_uploaded=False));b.emit('v62_export',sha256=b.sha(file),changed_tasks=changed)

def main():
    arg=argparse.ArgumentParser();arg.add_argument('stage',choices=['all','evaluate','fresh','export']);a=arg.parse_args();b.remote_only();torch.backends.cuda.matmul.allow_tf32=False;OUT.mkdir(exist_ok=True);d=b.build_data();g=graph.graphs(d);selftest(d,g)
    if a.stage in ['all','evaluate']:evaluate(d,g)
    if a.stage in ['all','fresh']:fresh(d,g,json.loads((OUT/'analysis.json').read_text())['tasks'])
    if a.stage in ['all','export']:export(d,g,json.loads((OUT/'fresh.json').read_text())['tasks'])
    b.save(OUT/(a.stage+'_completed.json'),dict(status='COMPLETE',signature=signature()))

if __name__=='__main__':main()
