"""v45 aggressive remote ensemble search from frozen v27/v44 predictions."""
import csv, hashlib, io, json
from pathlib import Path
import numpy as np
import admet_v24_models as b
import admet_v44_run as v44
import admet_v27_models as v27

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'admet_model_outputs_v45c'
P = {
    'version': 45,
    'baseline': 'v44',
    'baseline_score': 50.582708,
    'leader_score': 52.723518,
    'remote_root': '/home/czp123/admet_v23_20260928',
    'weights': [0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.35,0.50,0.65,0.80,1.0],
    'objective_valid_weight': 0.20,
    'objective_cv_weight': 0.80,
}

def signature():
    names=['admet_v45_search.py','admet_v44_run.py','admet_v44_protocol.json','admet_v27_models.py','admet_v27_protocol.json']
    return hashlib.sha256(json.dumps(dict(files={n:b.sha(ROOT/n) for n in names}, v44=v44.signature()),sort_keys=True).encode()).hexdigest()

def align_prediction(z, query, rows):
    q=np.asarray(z['query'],int); rows=np.asarray(rows,int)
    pos=np.searchsorted(q,rows)
    assert np.array_equal(q[pos],rows)
    return np.asarray(z['p'])[:,pos]

def load_v27(d):
    valid={}; cv={}; test={}
    folds=np.load(b.OUT/'folds.npy')
    cvbase=[]
    for k,t in enumerate(d['tasks']):
        vi=np.flatnonzero((d['tid']==k)&(d['split']==1)); tr=np.flatnonzero((d['tid']==k)&(d['split']==0)); te=np.flatnonzero((d['tid']==k)&(d['split']==2))
        # Use the persisted two-seed v27 validation average, rather than a
        # single-seed proxy, for the final task-level choice.
        zv=np.load(v27.OUT/'valid_predictions.npy')
        valid[t]=zv[:,vi]
        zc=np.load(v27.OUT/'cv_predictions.npy')[:,tr]
        cv[t]=zc
        parts=[]
        for seed in [270601,270602,270603]:
            z=np.load(v27.OUT/'final'/str(seed)/'predictions.npz')
            parts.append(align_prediction(z,te,te))
        test[t]=np.mean(parts,axis=0)
    return valid,cv,test

def evaluate(d):
    assert ROOT == Path(P['remote_root'])
    OUT.mkdir(parents=True,exist_ok=True)
    basev=np.load(v44.OUT/'baseline_valid.npy')
    basec=np.load(v44.OUT/'baseline_cv_proxy.npy')
    folds=np.load(b.OUT/'folds.npy')
    vv,cc,tt=load_v27(d)
    records=[]
    for k,t in enumerate(d['tasks']):
        m=d['metrics'][k]; vi=np.flatnonzero((d['tid']==k)&(d['split']==1)); tr=np.flatnonzero((d['tid']==k)&(d['split']==0))
        yv=d['y'][vi]; ytr=d['y'][tr]
        best=None; grid=[]
        for branch in range(vv[t].shape[0]):
            for w in P['weights']:
                newv=b.blend(m,basev[vi],vv[t][branch],w)
                vg=b.delta(m,yv,basev[vi],newv)
                gs=[]
                for fold in range(3):
                    ix=tr[folds[tr]==fold]
                    newc=b.blend(m,basec[ix],cc[t][branch,folds[tr]==fold],w)
                    gs.append(b.delta(m,d['y'][ix],basec[ix],newc))
                cvm=float(np.mean(gs)); cvs=float(np.std(gs)); obj=float(P['objective_valid_weight']*vg+P['objective_cv_weight']*cvm-0.04*cvs)
                row=dict(branch=branch,weight=w,valid_gain=float(vg),cv_gains=gs,cv_mean=cvm,cv_std=cvs,objective=obj)
                grid.append(row)
                if best is None or obj>best['objective']: best=row
        records.append(dict(task=t,metric=m,valid_rows=len(vi),selection=best,grid=grid,
            current_valid_metric=b.metric(m,yv,basev[vi]),candidate_valid_metric=b.metric(m,yv,b.blend(m,basev[vi],vv[t][best['branch']],best['weight']))))
        b.emit('v45_search',task=t,best=best)
    b.save(OUT/'analysis.json',dict(status='EVALUATED',signature=signature(),baseline='v44',baseline_score=P['baseline_score'],tasks=records,limitations='v27 predictions were fit before v44; selection uses historical valid and scaffold CV proxies; official weights unknown.'))
    return records,tt

def export(d,records,tt):
    source=next((ROOT/'admet_submissions/v44').glob('*.csv'))
    def read(path):return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))
    rows=read(source); template=read(ROOT/'admet_official/submission_template.csv'); changed={}
    for r in records:
        s=r['selection']; t=r['task']; m=r['metric']; k=d['tasks'].index(t)
        # Require useful scaffold-CV signal; valid is only a light tie-breaker by design.
        if s['cv_mean'] <= 0.0005 or s['objective'] <= 0.0005: continue
        pos=[i for i,row in enumerate(rows) if i>0 and row[0]==t]
        old=np.array([float(rows[i][4]) for i in pos]); p=tt[t][s['branch']]
        merged=b.blend(m,old,p,s['weight'])
        for i,x in zip(pos,merged):rows[i][4]=repr(float(x))
        changed[t]=dict(branch=s['branch'],weight=s['weight'],cv_mean=s['cv_mean'],objective=s['objective'],rows=int(np.sum(old!=merged)))
    assert changed and len(rows)==len(template)==16380 and rows[0]==template[0]
    for row,trow in zip(rows[1:],template[1:]):
        assert len(row)==6 and row[:4]+row[5:]==trow[:4]+trow[5:] and np.isfinite(float(row[4]))
        if row[5] in ['AUROC','AUPRC']: assert 0<=float(row[4])<=1
    dest=ROOT/'admet_submissions/v45c'; dest.mkdir(parents=True,exist_ok=True)
    path=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261014-v45.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows)
    path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'))
    assert read(path)==rows
    sha=b.sha(path)
    b.save(OUT/'export.json',dict(status='EXPERIMENTAL_AWAITING_PLATFORM',signature=signature(),baseline='v44',baseline_score=P['baseline_score'],sha256=sha,rows=16379,columns=6,changed_tasks=changed,bytes=path.stat().st_size,encoding='UTF-8 BOM',line_endings='CRLF',template_identity_exact=True,automatically_uploaded=False,local_gpu_training=False,training_host='czp123@100.109.143.42',platform_score=None))
    return path

def main():
    b.remote_only()
    d=b.build_data(); records,tt=evaluate(d); path=export(d,records,tt); print(path); print(json.dumps({r['task']:r['selection'] for r in records},ensure_ascii=False,indent=2))
if __name__=='__main__': main()





