"""Freeze one combined release after source training and independent reloads complete."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse,json,csv,io,hashlib
from pathlib import Path
import numpy as np
import admet_v24_models as b
import admet_v61_run as v61
import admet_v62_run as v62
import admet_v63_run as v63
import admet_v65_run as v65
import admet_v60_baseline as anchor
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v64';P=json.loads((ROOT/'admet_v64_protocol.json').read_text())
MODULES={61:v61,62:v62,63:v63,65:v65}

def signature():return hashlib.sha256((b.sha(__file__)+b.sha(ROOT/'admet_v64_protocol.json')+''.join(MODULES[x].signature() for x in P['source_versions'])).encode()).hexdigest()

def values(d,version,task,s,mode,fold=None):
    k=d['tasks'].index(task);m=d['metrics'][k];source=ROOT/f'admet_model_outputs_v{version}'
    if mode=='cv':
        r=np.flatnonzero((d['tid']==k)&(d['split']==0));f=np.load(b.OUT/'folds.npy');r=r[f[r]==fold]
        raw=np.load(source/'cv_predictions.npy')[:,r] if version==62 else np.load(source/task/'cv_predictions.npy')[:,r]
    else:
        r=np.flatnonzero((d['tid']==k)&(d['split']==1))
        if version==62:raw=np.load(source/'valid_predictions.npy')[:,r]
        else:
            z=np.load(source/task/'valid/predictions.npz');np.testing.assert_array_equal(z['query'],r);raw=z['p']
    p=v62.variants(raw,m)[s['variant']] if version==62 else raw[s['variant']]
    assert np.isfinite(p).all();return r,p

def main():
    b.remote_only();OUT.mkdir(exist_ok=True);d=b.build_data();bv,bc=v61.baseline(d);f=np.load(b.OUT/'folds.npy');sources={};replays={}
    for version in P['source_versions']:
        mod=MODULES[version];report=json.loads((mod.OUT/'fresh.json').read_text());assert report['status']=='COMPLETE' and report['signature']==mod.signature()
        chosen=[r for r in report['tasks'] if r['pass_gate']]
        if version==61:
            extra=json.loads((mod.OUT/'independent_chemical_gate.json').read_text());assert extra['status']=='COMPLETE' and extra['signature']==mod.signature()
            failed={r['task'] for r in extra['tasks'] if not r['pass_gate']};chosen=[r for r in chosen if r['task'] not in failed]
        if chosen:
            check=json.loads((mod.OUT/'model_replay.json').read_text());assert check['status']=='PASS';replays[version]=json.loads((mod.OUT/'fusion_replay.json').read_text())
        for r in chosen:sources.setdefault(r['task'],[]).append(dict(version=version,record=r))
    rows=anchor.rows(next((ROOT/'admet_submissions/v60').glob('*.csv')));assert b.sha(next((ROOT/'admet_submissions/v60').glob('*.csv')))==P['baseline_sha256'];template=anchor.rows(ROOT/'admet_official/submission_template.csv');replay={};chosen={};diagnostics=[];changed={}
    for task,options in sources.items():
        k=d['tasks'].index(task);m=d['metrics'][k];grid=[]
        for opt in options:
            version=opt['version'];s=opt['record']['selection'];gains=[]
            for fold in range(3):
                ix,p=values(d,version,task,s,'cv',fold);gains.append(b.delta(m,d['y'][ix],bc[ix],b.blend(m,bc[ix],p,s['weight'])))
            vi,p=values(d,version,task,s,'valid');vg=b.delta(m,d['y'][vi],bv[vi],b.blend(m,bv[vi],p,s['weight']))
            np.testing.assert_allclose(gains,s['cv_gains'],atol=1e-8,rtol=1e-7)
            grid.append(dict(version=version,selection=s,cv_gains=gains,cv_mean=float(np.mean(gains)),objective=float(np.mean(gains)-.1*np.std(gains)),valid_gain=vg))
        held=[max(grid,key=lambda x:np.mean([v for j,v in enumerate(x['cv_gains']) if j!=fold]))['cv_gains'][fold] for fold in range(3)]
        best=max(grid,key=lambda x:x['objective']) if np.mean(held)>.001 else min(grid,key=lambda x:x['version']);version=best['version'];r=replays[version][task];pos=[i for i,a in enumerate(rows) if i and a[0]==task];old=np.array([float(rows[i][4]) for i in pos]);np.testing.assert_array_equal(old,r['baseline']);new=np.array(r['result']);assert len(new)==len(old)
        for i,value in zip(pos,new):rows[i][4]=repr(float(value))
        changed[task]=int(np.sum(new!=old));replay[task]=dict(**r,source_version=version);chosen[task]=best;diagnostics.append(dict(task=task,choices=grid,held_selection_gains=held,held_mean=float(np.mean(held)),selected=version));b.emit('v64_choice',task=task,source=version,cv_gain=best['cv_mean'])
    assert changed and len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,t in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==t[:4]+t[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v64';dest.mkdir(parents=True,exist_ok=True);file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261017.csv';buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows);raw=b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8')
    if file.exists():assert file.read_bytes()==raw
    else:file.write_bytes(raw)
    b.save(OUT/'fusion_replay.json',replay);b.save(OUT/'analysis.json',dict(status='FROZEN',signature=signature(),tasks=diagnostics));b.save(OUT/'release.json',dict(status='AWAITING_PLATFORM',signature=signature(),selected=chosen,sha256=b.sha(file),official_score=None,limitations=P['limitations']));b.save(OUT/'model_replay.json',dict(status='PASS',source_checks={str(x):json.loads((ROOT/f'admet_model_outputs_v{x}/model_replay.json').read_text()) for x in replays},source_model_replay_passed=True,no_additional_model_fitting=True));b.save(dest/'verification.json',dict(status='AWAITING_PLATFORM_EVALUATION',signature=signature(),sha256=b.sha(file),baseline_sha256=P['baseline_sha256'],rows=16379,columns=6,tasks=22,changed_tasks=changed,bytes=len(raw),encoding='UTF-8 BOM',line_endings='CRLF',official_score=None,known_score_upper_bound=50.804792,local_gpu_training=False,automatically_uploaded=False));b.emit('v64_export',sha256=b.sha(file),changed_tasks=changed)

if __name__=='__main__':main()
