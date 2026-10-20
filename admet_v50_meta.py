"""v50 remote OOF meta-fusion over correctly expanded historical variants."""
import csv,io,json,hashlib
from pathlib import Path
import numpy as np
import admet_v24_models as b
import admet_v48_fusion as hist
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'admet_model_outputs_v50'
VERSIONS=hist.VERSIONS
ALPHAS=[.02,.03,.05,.08,.12,.18,.25,.35,.5,.7,1.0]

def sig():
    return hashlib.sha256(json.dumps({'meta':b.sha(ROOT/'admet_v50_meta.py'),'hist':b.sha(ROOT/'admet_v48_fusion.py')},sort_keys=True).encode()).hexdigest()

def trans(m,p):
    p=np.asarray(p,float)
    if p.ndim==1: p=p[None]
    return p if m=='MAE' else np.stack([b.rank(x) for x in p])

def eval_mix(m,y,base,cand,weights,alpha):
    bt=trans(m,base); ct=trans(m,cand)
    if bt.ndim==2 and bt.shape[0]==1: bt=bt[0]
    if m=='MAE': pred=(1-alpha)*bt+alpha*np.tensordot(weights,ct,axes=(0,0))
    else: pred=(1-alpha)*bt+alpha*np.tensordot(weights,ct,axes=(0,0))
    return float(b.delta(m,y,base,pred)),pred

def main():
    b.remote_only();OUT.mkdir(exist_ok=True);d=b.build_data();folds=np.load(ROOT/'admet_model_outputs_v24/folds.npy')
    basev,basec,baset=hist.current_v46(d); compact_paths={v:ROOT/f'admet_model_outputs_v{v}/test_predictions_compact.json' for v in VERSIONS}
    records=[];chosen={};out_test={}
    for k,t in enumerate(d['tasks']):
        m=d['metrics'][k];vi=np.flatnonzero((d['tid']==k)&(d['split']==1));tr=np.flatnonzero((d['tid']==k)&(d['split']==0));te=np.flatnonzero((d['tid']==k)&(d['split']==2));trc=tr[np.isfinite(basec[tr])]
        src=[]
        for v in VERSIONS:
            vp,cp,tp=hist.load_source(ROOT/f'admet_model_outputs_v{v}',v,t,m,vi,trc,te,compact_paths[v])
            if vp is None or cp is None or tp is None:continue
            for j in range(vp.shape[0]):
                if np.isfinite(vp[j]).all() and np.isfinite(cp[j]).all() and np.isfinite(tp[j]).all():src.append(dict(id=f'v{v}:{j}',v=vp[j],c=cp[j],t=tp[j]))
        singles=[]
        for si,s in enumerate(src):
            gs=[]
            for fold in range(3):
                ix=trc[folds[trc]==fold];loc=np.flatnonzero(folds[trc]==fold)
                if len(ix):gs.append(eval_mix(m,d['y'][ix],basec[ix],s['c'][loc],np.array([1.]),1.)[0])
            singles.append((float(np.mean(gs)),si,gs))
        top=[x[1] for x in sorted(singles,reverse=True)[:min(8,len(singles))]]
        rng=np.random.default_rng(50000+k); weight_sets=[]
        for si in top: 
            w=np.zeros(len(top));w[top.index(si)]=1.;weight_sets.append(w)
        for _ in range(2400):
            z=rng.dirichlet(np.ones(len(top))*.55);weight_sets.append(z)
        best=None
        for w in weight_sets:
            for alpha in ALPHAS:
                gs=[]
                for fold in range(3):
                    ix=trc[folds[trc]==fold];loc=np.flatnonzero(folds[trc]==fold)
                    if not len(ix):continue
                    cand=np.stack([src[si]['c'][loc] for si in top])
                    gs.append(eval_mix(m,d['y'][ix],basec[ix],cand,w,alpha)[0])
                if not gs:continue
                cm=float(np.mean(gs)); obj=cm-.04*float(np.std(gs))
                valid_cand=np.stack([src[si]['v'] for si in top]); vg=eval_mix(m,d['y'][vi],basev[vi],valid_cand,w,alpha)[0]
                # Use valid only as a mild guard; OOF CV is the main objective.
                score=obj+.08*max(vg,-.01)
                row=dict(alpha=float(alpha),weights=w.tolist(),top=top,cv_gains=gs,cv_mean=cm,objective=float(obj),valid_gain=float(vg),score=float(score))
                if sum(x>0 for x in gs)>=2 and (best is None or score>best['score']):best=row
        rec=dict(task=t,metric=m,sources=len(src),top_ids=[src[i]['id'] for i in top],selection=None if best is None else {**best,'top_ids':[src[i]['id'] for i in top]})
        # Keep only candidates with positive OOF objective and avoid severe valid collapse.
        if best is not None and best['objective']>0.0004 and best['valid_gain']>-0.015:
            cand=np.stack([src[si]['t'] for si in top]);tt=trans(m,cand);mix=np.tensordot(np.asarray(best['weights']),tt,axes=(0,0));bt=trans(m,baset[t]);out_test[t]=(1-best['alpha'])*bt+best['alpha']*mix
            chosen[t]={**best,'top_ids':[src[i]['id'] for i in top]}
        else:
            out_test[t]=baset[t];rec['selection']=None
        records.append(rec);b.emit('v50_meta',task=t,sources=len(src),selection=rec['selection'])
    b.save(OUT/'meta.json',dict(signature=sig(),status='FROZEN',baseline='v46',baseline_score=50.692048,tasks=records,selected_tasks=list(chosen),chosen=chosen,limitations='OOF meta-fusion over historical predictions; official weights unknown.'))
    basefile=next((ROOT/'admet_submissions/v46').glob('*.csv'));rows=list(csv.reader(io.StringIO(basefile.read_bytes().decode('utf-8-sig'),newline='')));temp=list(csv.reader(io.StringIO((ROOT/'admet_official/submission_template.csv').read_bytes().decode('utf-8-sig'),newline='')))
    for i,row in enumerate(rows):
        if i==0:continue
        t=row[0];si=int(row[1]);
        if t in out_test:row[4]=repr(float(out_test[t][si]))
    assert len(rows)==len(temp)==16380 and rows[0]==temp[0]
    for a,z in zip(rows[1:],temp[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v50';dest.mkdir(exist_ok=True);path=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261015-v50.csv';buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows);path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode())
    b.save(OUT/'export.json',dict(status='EXPERIMENTAL_AWAITING_PLATFORM',signature=sig(),baseline='v46',baseline_score=50.692048,sha256=b.sha(path),rows=16379,columns=6,changed_tasks=list(chosen),changed_rows=int(sum(1 for i in range(1,len(rows)) if rows[i][4]!=basefile.read_text(encoding='utf-8-sig').splitlines()[i].split(',')[4] if False)),automatically_uploaded=False,local_gpu_training=False,training_host='czp123@100.109.143.42',platform_score=None))
    print(json.dumps({'path':str(path),'sha256':b.sha(path),'selected_tasks':list(chosen)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
