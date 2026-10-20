"""Remote primary-metric ensemble optimization under the user's new policy."""
import argparse,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from admet_v6_runner import remote_only,save,sha,get_metric,metrics,kind,partition,emit
from admet_v15_models import ALLOWED,load_task,fit,predict
from admet_v15_ensemble import combine,score,select

OUT=Path('admet_model_outputs_v15')
ROOT=Path('admet_release_v15')
PFILE=Path('admet_v15_protocol.json')


def read(p):return json.loads(Path(p).read_text())
def protocol():return read(PFILE)
def names():return read('admet_v13_protocol.json')['tasks']


def anchor(task):
    b=pd.read_csv(Path('admet_model_outputs_v12')/task/'baseline_oof.csv')
    route=read(Path('admet_release_v14')/task/'selection.json');v=route['version']
    if v!=11:
        source=Path(f'admet_model_outputs_v{v}')/task;s=read(source/'decision.json')['selection']
        c=pd.read_csv(source/(s['family']+'_oof.csv'))
        np.testing.assert_array_equal(c.row,b.row);np.testing.assert_array_equal(c.fold,b.fold);np.testing.assert_allclose(c.Y_true,b.Y_true)
        b['Y_pred']=(1-s['alpha'])*b.Y_pred+s['alpha']*c.Y_pred
    return b


def candidates(task,b):
    result={};provenance=[]
    for v,fs in ALLOWED.items():
        for family in fs:
            path=Path(f'admet_model_outputs_v{v}')/task/(family+'_oof.csv')
            if not path.exists():continue
            c=pd.read_csv(path)
            np.testing.assert_array_equal(c.row,b.row);np.testing.assert_array_equal(c.fold,b.fold);np.testing.assert_allclose(c.Y_true,b.Y_true)
            q=c.Y_pred.to_numpy(float);assert np.isfinite(q).all()
            name=f'v{v}:{family}';result[name]=q
            provenance.append(dict(name=name,path=str(path),sha256=sha(path),recipe_sha256=sha(f'admet_v{v}_protocol.json')))
    assert result
    return result,provenance


def shortlist(task):
    dest=OUT/task;dest.mkdir(parents=True,exist_ok=True)
    locked=dest/'shortlist.json'
    if locked.exists():return read(locked)
    p=protocol();b=anchor(task);pool,provenance=candidates(task,b);metric=get_metric(task)
    y=b.Y_true.to_numpy();base=b.Y_pred.to_numpy();fold=b.fold.to_numpy();rows=[]
    for name,c in pool.items():
        gains=[score(y[fold==f],base[fold==f],.5*base[fold==f]+.5*c[fold==f],metric) for f in range(3)]
        rows.append(dict(name=name,mean_gain=float(np.mean(gains)),fold_gains=gains))
    rows.sort(key=lambda r:(-r['mean_gain'],r['name']));selected=[rows[0]['name']]
    eligible=[r['name'] for r in rows[1:] if abs(spearmanr(pool[selected[0]],pool[r['name']]).statistic)<.995]
    if len(rows)>1:selected.append(eligible[0] if eligible else rows[1]['name'])
    result=dict(task=task,selected=selected,ranked=rows,provenance=provenance,protocol_sha256=sha(PFILE),
        selection_data='Previously reused train scaffold OOF only',v14_baseline_reconstructed=True)
    b.to_csv(dest/'baseline_oof.csv',index=False);save(locked,result)
    emit(stage='v15_shortlist',task=task,selected=selected)
    return result


def materialize(task,d,ix,query,names,stage):
    dest=OUT/task/stage;dest.mkdir(parents=True,exist_ok=True);preds=[]
    frame=pd.concat([d['frames']['train'],d['frames']['valid']],ignore_index=True)
    assert len(ix)>0 and np.min(ix)>=0 and np.max(ix)<len(frame)
    y=frame.Y.to_numpy(float)[ix]
    for name in names:
        key=name.replace(':','_');path=dest/(key+'.joblib');meta=dest/(key+'.json');cache=dest/(key+'.npz')
        if path.exists() and meta.exists() and cache.exists():
            m=read(meta);assert m['protocol_sha256']==sha(PFILE) and m['sha256']==sha(path)
            with np.load(cache) as z:
                np.testing.assert_array_equal(z['fit'],ix);np.testing.assert_array_equal(z['query'],query);q=z['prediction'].copy()
        else:
            start=time.time();bundle=fit(name,d,ix,y,get_metric(task))
            np.testing.assert_array_equal(bundle['fit_rows'],ix)
            assert not set(query)&set(ix)
            for audit in bundle['model'].get('source_audit',[]):
                assert audit['heldout_scaffold_overlap']==audit['heldout_molecule_overlap']==0
            q=predict(bundle,d,query);joblib.dump(bundle,path,compress=3)
            np.savez_compressed(cache,fit=ix,query=query,prediction=q)
            save(meta,dict(name=name,sha256=sha(path),protocol_sha256=sha(PFILE),fit_rows=len(ix),query_rows=len(query),
                training_seconds=time.time()-start,source_audit=bundle['model'].get('source_audit',[])))
            del bundle
            emit(stage='v15_fit',task=task,partition=stage,name=name,seconds=round(time.time()-start))
        preds.append(q)
    return preds


def diagnostics(task,split,d,it,iv,y,b,q):
    from admet_v6_domain_guard import nearest
    metric=get_metric(task);out=[]
    for label,mask in [('all',np.ones(len(iv),bool)),('distant',nearest(d['raw'],it,iv)<.3)]:
        covered=mask.sum()>=2 and (kind(metric)!='classification' or len(np.unique(y[mask]))==2)
        row=dict(task=task,split=split,group=label,n=int(mask.sum()),coverage_sufficient=bool(covered),used_as_veto=False)
        if covered:
            before=metrics(kind(metric),y[mask],b[mask]);after=metrics(kind(metric),y[mask],q[mask]);row.update(before=before,after=after,primary_gain=score(y[mask],b[mask],q[mask],metric))
        out.append(row)
    return out


def evaluate(task):
    dest=OUT/task;decision=dest/'decision.json'
    if decision.exists():assert read(decision)['protocol_sha256']==sha(PFILE);return
    p=protocol();s=shortlist(task);b=anchor(task);pool,_=candidates(task,b)
    chosen=s['selected'];d=load_task(task);n=len(d['frames']['train']);nv=len(d['frames']['valid']);metric=get_metric(task)
    np.testing.assert_array_equal(b.fold,partition(task,d));np.testing.assert_allclose(b.Y_true,d['frames']['train'].Y)
    y=b.Y_true.to_numpy();base=b.Y_pred.to_numpy();fold=b.fold.to_numpy();cc=[pool[name] for name in chosen]
    it=np.arange(n);iv=np.arange(n,n+nv);yv=d['frames']['valid'].Y.to_numpy(float)
    bv=pd.read_csv(Path('admet_release_v14')/task/'validation_predictions.csv');np.testing.assert_allclose(bv.Y_true,yv);bv=bv.Y_pred.to_numpy()
    cv=materialize(task,d,it,iv,chosen,'valid')
    # All 22 shortlists are locked before any new official-valid evaluation.
    assert len(list(OUT.glob('*/shortlist.json')))==22
    if (dest/'selection.json').exists():best=read(dest/'selection.json')
    else:
        best,grid=select(y,fold,base,cc,yv,bv,cv,metric,p)
        best.update(names=chosen,metric=metric,protocol_sha256=sha(PFILE))
        save(dest/'selection_grid.json',grid);save(dest/'selection.json',best)
    q=np.empty(n);cross=np.empty(n);diag=[]
    for f in range(3):
        mask=fold==f
        q[mask]=combine(base[mask],[c[mask] for c in cc],best['weights'],best['mode'],metric)
        # Report only: cached base OOF predictions do not make this nested independent CV.
        loo,_=select(y,fold,base,cc,yv,bv,cv,metric,p,fold_subset=[x for x in range(3) if x!=f])
        cross[mask]=combine(base[mask],[c[mask] for c in cc],loo['weights'],loo['mode'],metric)
        diag+=diagnostics(task,f'fold_{f}',d,np.flatnonzero(~mask),np.flatnonzero(mask),y[mask],base[mask],q[mask])
    qv=combine(bv,cv,best['weights'],best['mode'],metric)
    diag+=diagnostics(task,'valid',d,it,iv,yv,bv,qv)
    pd.DataFrame(dict(row=b.row,fold=fold,Y_true=y,baseline=base,blend=q,cross_meta_diagnostic=cross)).to_csv(dest/'oof_comparison.csv',index=False)
    pd.DataFrame(dict(Y_true=yv,baseline=bv,blend=qv,**{f'candidate_{i}':c for i,c in enumerate(cv)})).to_csv(dest/'validation_comparison.csv',index=False)
    changed=best['candidate_mass']>0
    repeat={}
    if changed:
        it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=692509).split(np.arange(n),groups=d['groups'][:n]))
        from admet_v12_runner import repeat_baseline
        route=read(Path('admet_release_v14')/task/'selection.json')
        br=pd.read_csv(Path(f'admet_model_outputs_v{route["version"]}')/task/'repeat_comparison.csv').blend.to_numpy() if route['version']!=11 else repeat_baseline(task,d,it,iv)
        active=[j for j,w in enumerate(best['weights'][1:]) if w>0];cr=[br.copy() for _ in chosen]
        fitted=materialize(task,d,it,iv,[chosen[j] for j in active],'repeat')
        for j,c in zip(active,fitted):cr[j]=c
        qr=combine(br,cr,best['weights'],best['mode'],metric)
        repeat=dict(before=metrics(kind(metric),y[iv],br),after=metrics(kind(metric),y[iv],qr),primary_gain=score(y[iv],br,qr,metric),used_as_veto=False)
        pd.DataFrame(dict(row=iv,Y_true=y[iv],baseline=br,blend=qr)).to_csv(dest/'repeat_comparison.csv',index=False)
        diag+=diagnostics(task,'repeat',d,it,iv,y[iv],br,qr)
    result=dict(task=task,changed=changed,selection=best,before=metrics(kind(metric),yv,bv),after=metrics(kind(metric),yv,qv),
        repeat=repeat,diagnostics=diag,protocol_sha256=sha(PFILE),per_metric_vetoes=False,
        interpretation='Selection-biased primary-metric surrogate; not a platform score forecast')
    save(decision,result);emit(stage='v15_decision',task=task,changed=changed,weights=best['weights'],mode=best['mode'],objective=best['objective'])


if __name__=='__main__':
    remote_only();ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['shortlist','evaluate']);ap.add_argument('--tasks',nargs='*');args=ap.parse_args()
    with threadpool_limits(limits=4):
        for t in args.tasks or names():
            (shortlist if args.stage=='shortlist' else evaluate)(t)
