"""Remote historical candidate fusion for v47."""
import csv, io, json, hashlib, re
from pathlib import Path
import numpy as np
import admet_v24_models as b

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_model_outputs_v47'
VERSIONS=['24','27','29','31','32','33','35','37','40','41','43','44','46']
WEIGHTS=[0.01,0.02,0.03,0.05,0.08,0.12,0.18,0.25,0.35,0.50,0.70,1.0]
PAIR_ALPHA=[0.25,0.5,0.75]

def sig():
    names=['admet_v47_fusion.py','admet_v24_models.py','admet_v46_protocol.json']
    return hashlib.sha256(json.dumps({x:b.sha(ROOT/x) for x in names},sort_keys=True).encode()).hexdigest()

def arr(v):
    if isinstance(v,dict):
        if 'predictions' in v:return np.asarray(v['predictions'],dtype=float)
        # compact v37 style numeric keys
        try:return np.asarray([v[k] for k in sorted(v,key=lambda x:int(x))],dtype=float)
        except Exception:return None
    try:
        a=np.asarray(v,dtype=float)
        if a.ndim==1:return a[None]
        if a.ndim==2:return a
    except Exception:pass
    return None

def load_compact(path,task,n):
    if not path.exists():return None
    try:
        x=json.loads(path.read_text())
        if task not in x:return None
        a=arr(x[task])
        if a is not None and a.shape[-1]==n:return a
    except Exception:pass
    return None

def align_npz(path, rows):
    if not path.exists():return None
    try:
        z=np.load(path,allow_pickle=True)
        if 'p' not in z or 'query' not in z:return None
        q=np.asarray(z['query'],int); rows=np.asarray(rows,int)
        if len(q)!=len(rows) or set(q.tolist())!=set(rows.tolist()):return None
        pos=np.searchsorted(q,rows)
        if not np.array_equal(q[pos],rows):return None
        p=np.asarray(z['p'],float)
        if p.ndim==1:p=p[None]
        return p[:,pos]
    except Exception:return None

def load_source(out,task,vi,tr,te,compact):
    vp=align_npz(out/task/'valid/predictions.npz',vi)
    cp=None
    p=out/task/'cv_predictions.npy'
    if p.exists():
        try:
            a=np.asarray(np.load(p,allow_pickle=True),float)
            if a.ndim==1:a=a[None]
            if a.ndim==2 and a.shape[1]>=max(tr)+1:
                z=a[:,tr]
                if np.isfinite(z).all(): cp=z
        except Exception:pass
    tp=align_npz(out/task/'final/predictions.npz',te)
    if tp is None: tp=load_compact(compact,task,len(te))
    return vp,cp,tp

def current_v46(d):
    basev=np.load(ROOT/'admet_model_outputs_v46/baseline_valid.npy').copy()
    basec=np.load(ROOT/'admet_model_outputs_v46/baseline_cv_proxy.npy').copy()
    report=json.loads((ROOT/'admet_model_outputs_v46/analysis.json').read_text())
    for r in report['tasks']:
        s=r.get('selection')
        if not s:continue
        k=d['tasks'].index(r['task']); m=d['metrics'][k]
        vi=np.flatnonzero((d['tid']==k)&(d['split']==1)); tr=np.flatnonzero((d['tid']==k)&(d['split']==0))
        vp=np.load(ROOT/'admet_model_outputs_v46'/r['task']/'valid_predictions.npy')[s['variant']]
        cp=np.load(ROOT/'admet_model_outputs_v46'/r['task']/'cv_predictions.npy')[s['variant']]
        basev[vi]=b.blend(m,basev[vi],vp,s['weight'])
        basec[tr]=b.blend(m,basec[tr],cp[tr],s['weight'])
    # Test baseline: replay the exact v46 export on baseline_v45.csv.
    source=ROOT/'admet_model_outputs_v46/baseline_v45.csv'
    rows=list(csv.reader(io.StringIO(source.read_bytes().decode('utf-8-sig'),newline='')))
    test={}
    compact=json.loads((ROOT/'admet_model_outputs_v46/test_predictions_compact.json').read_text())
    for r in report['tasks']:
        s=r.get('selection')
        if not s:continue
        k=d['tasks'].index(r['task']);m=d['metrics'][k];t=r['task']
        p=np.asarray(compact[t][s['variant']],float)
        pos=[i for i,row in enumerate(rows) if i>0 and row[0]==t]
        old=np.asarray([float(rows[i][4]) for i in pos])
        test[t]=b.blend(m,old,p,s['weight'])
    for k,t in enumerate(d['tasks']):
        if t not in test:
            pos=np.flatnonzero((d['tid']==k)&(d['split']==2))
            test[t]=np.asarray([float(rows[i+1][4]) for i in range(len(pos))]) if False else None
    # Fill unchanged tasks from baseline csv in template task order.
    for k,t in enumerate(d['tasks']):
        if test.get(t) is None:
            pos=[i for i,row in enumerate(rows) if i>0 and row[0]==t]
            test[t]=np.asarray([float(rows[i][4]) for i in pos])
    return basev,basec,test

def metric_safe(m,y,p):
    try:return b.metric(m,y,p)
    except Exception:return np.nan

def candidate_test_vector(tp, idx):
    if tp is None:return None
    if tp.ndim==1:tp=tp[None]
    if idx<tp.shape[0]:return tp[idx]
    return np.mean(tp,axis=0)

def main():
    b.remote_only(); OUT.mkdir(exist_ok=True)
    d=b.build_data(); folds=np.load(ROOT/'admet_model_outputs_v24/folds.npy')
    basev,basec,baset=current_v46(d)
    compact_paths={v:ROOT/f'admet_model_outputs_v{v}/test_predictions_compact.json' for v in VERSIONS}
    allrec=[]; chosen={}; test_out={}
    for k,t in enumerate(d['tasks']):
        m=d['metrics'][k]; vi=np.flatnonzero((d['tid']==k)&(d['split']==1)); tr=np.flatnonzero((d['tid']==k)&(d['split']==0)); te=np.flatnonzero((d['tid']==k)&(d['split']==2))
        mask=np.isfinite(basec[tr]); trc=tr[mask]
        sources=[]
        for v in VERSIONS:
            vp,cp,tp=load_source(ROOT/f'admet_model_outputs_v{v}',t,vi,trc,te,compact_paths[v])
            if vp is None or cp is None or tp is None:continue
            if vp.shape[1]!=len(vi) or cp.shape[1]!=len(trc) or tp.shape[1]!=len(te):continue
            n=min(vp.shape[0],cp.shape[0],tp.shape[0])
            for j in range(n):
                if not (np.isfinite(vp[j]).all() and np.isfinite(cp[j]).all() and np.isfinite(tp[j]).all()):continue
                sources.append(dict(id=f'v{v}:{j}',version=v,index=j,v=vp[j],c=cp[j],t=tp[j]))
        # score single candidates, then pair only among top candidates.
        grid=[]
        for si,s in enumerate(sources):
            for w in WEIGHTS:
                vg=b.delta(m,d['y'][vi],basev[vi],b.blend(m,basev[vi],s['v'],w))
                gs=[]
                for f in range(3):
                    ix=trc[folds[trc]==f]
                    if len(ix)==0:continue
                    loc=np.flatnonzero(folds[trc]==f)
                    new=b.blend(m,basec[ix],s['c'][loc],w)
                    gs.append(b.delta(m,d['y'][ix],basec[ix],new))
                cm=float(np.mean(gs)) if gs else -1
                obj=.15*vg+.85*cm-.04*float(np.std(gs))
                grid.append(dict(kind='single',a=si,b=None,alpha=1.0,weight=w,valid_gain=float(vg),cv_gains=gs,cv_mean=cm,objective=float(obj)))
        top=sorted(grid,key=lambda x:x['objective'],reverse=True)[:8]
        for ia in range(len(top)):
            for ib in range(ia+1,len(top)):
                a=sources[top[ia]['a']];bb=sources[top[ib]['a']]
                if a['id']==bb['id']:continue
                for al in PAIR_ALPHA:
                    vv=al*a['v']+(1-al)*bb['v']; cc=al*a['c']+(1-al)*bb['c']
                    for w in [0.03,0.05,0.08,0.12,0.18,0.25,0.35,0.5]:
                        vg=b.delta(m,d['y'][vi],basev[vi],b.blend(m,basev[vi],vv,w)); gs=[]
                        for f in range(3):
                            ix=trc[folds[trc]==f]; loc=np.flatnonzero(folds[trc]==f)
                            if len(ix):gs.append(b.delta(m,d['y'][ix],basec[ix],b.blend(m,basec[ix],cc[loc],w)))
                        cm=float(np.mean(gs));obj=.15*vg+.85*cm-.04*float(np.std(gs))
                        grid.append(dict(kind='pair',a=top[ia]['a'],b=top[ib]['a'],alpha=al,weight=w,valid_gain=float(vg),cv_gains=gs,cv_mean=cm,objective=float(obj)))
        # require a stable positive CV proxy and at least two positive folds; retain best.
        eligible=[x for x in grid if x['cv_mean']>0.0003 and x['objective']>0.0002 and sum(g>0 for g in x['cv_gains'])>=2]
        best=max(eligible,key=lambda x:x['objective']) if eligible else None
        rec=dict(task=t,metric=m,valid_rows=len(vi),train_rows=len(trc),source_count=len(sources),base_valid=metric_safe(m,d['y'][vi],basev[vi]),selection=None if best is None else {k:v for k,v in best.items() if k not in ['a','b']},top=[])
        for z in sorted(grid,key=lambda x:x['objective'],reverse=True)[:20]:
            q={k:v for k,v in z.items() if k not in ['a','b']}; q['a_id']=sources[z['a']]['id']; q['b_id']=sources[z['b']]['id'] if z['b'] is not None else None; rec['top'].append(q)
        if best is not None:
            a=sources[best['a']];
            if best['kind']=='single': cand=a['t']
            else:
                bb=sources[best['b']];cand=best['alpha']*a['t']+(1-best['alpha'])*bb['t']
            new=b.blend(m,baset[t],cand,best['weight']); test_out[t]=new
            chosen[t]=dict(**best,a_id=a['id'],b_id=sources[best['b']]['id'] if best['b'] is not None else None)
        else:test_out[t]=baset[t]
        allrec.append(rec); b.emit('v47_task',task=t,sources=len(sources),selection=rec['selection'])
    b.save(OUT/'fusion.json',dict(signature=sig(),status='FROZEN',baseline='v46',baseline_score=50.692048,tasks=allrec,selected_tasks=list(chosen),chosen=chosen,limitations='Historical candidate fusion scored with reused scaffold CV proxies; official task weights and score unavailable.'))
    # Export exact template clone from v46 submission when available, otherwise replay current baseline file.
    cur=ROOT/'admet_submissions/v46'
    files=list(cur.glob('*.csv'))
    if files:
        rows=list(csv.reader(io.StringIO(files[0].read_bytes().decode('utf-8-sig'),newline='')))
    else:
        rows=list(csv.reader(io.StringIO((ROOT/'admet_model_outputs_v46/baseline_v45.csv').read_bytes().decode('utf-8-sig'),newline='')))
    template=list(csv.reader(io.StringIO((ROOT/'admet_official/submission_template.csv').read_bytes().decode('utf-8-sig'),newline='')))
    for i,row in enumerate(rows):
        if i==0:continue
        t=row[0]; si=int(row[1]);
        if t in test_out: row[4]=repr(float(test_out[t][si]))
    assert len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,z in zip(rows[1:],template[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v47';dest.mkdir(exist_ok=True)
    path=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261015-v47.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows);path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode())
    b.save(OUT/'export.json',dict(status='EXPERIMENTAL_AWAITING_PLATFORM',signature=sig(),baseline='v46',baseline_score=50.692048,sha256=b.sha(path),rows=16379,columns=6,changed_tasks=list(chosen),changed_rows=int(sum(np.sum(np.asarray([float(rows[i][4]) for i in range(1,len(rows)) if rows[i][0]==t])!=np.asarray([float(x) for x in baset[t]])) for t in chosen)),automatically_uploaded=False,local_gpu_training=False,training_host='czp123@100.109.143.42',platform_score=None))
    print(json.dumps(dict(path=str(path),sha256=b.sha(path),selected=chosen),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
