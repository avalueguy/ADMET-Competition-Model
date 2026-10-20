"""Create versioned runners from v15; no historical files modified."""
from pathlib import Path
def write(name,text):
    p=Path(name);assert not p.exists(),name;p.write_text(text,encoding='utf-8')
s=Path('admet_v15_ensemble.py').read_text(encoding='utf-8').replace('from admet_v6_runner import','from itertools import product\nfrom admet_v6_runner import')
a=s.index('    if count==1:',s.index('def options'));b=s.index('\n\n\ndef score',a)
s=s[:a]+"    assert 1<=count<=3\n    return [[round(1-sum(x),10),*x] for x in product(grid,repeat=count) if sum(x)<=1+1e-9]\n"+s[b:]
write('admet_v16_ensemble.py',s)
s=Path('admet_v15_runner.py').read_text(encoding='utf-8').replace('v15','v16').replace('v14','v15')
s=s.replace('from admet_v16_models import ALLOWED,load_task,fit,predict','from admet_v16_models import load_task,fit,predict\nfrom admet_v15_runner import candidates as historical_candidates')
a=s.index('def anchor(task):');b=s.index('\n\ndef materialize',a)
s=s[:a]+'''def anchor(task):
    c=pd.read_csv(Path('admet_model_outputs_v15')/task/'oof_comparison.csv')
    return pd.DataFrame(dict(row=c.row,fold=c.fold,Y_true=c.Y_true,Y_pred=c.blend))


def candidates(task,b):
    selected=read(OUT/task/'shortlist.json');pool,provenance=historical_candidates(task,b)
    result={selected['historical']:pool[selected['historical']]}
    for family in protocol()['new_families']:
        path=OUT/task/(family+'_oof.csv');c=pd.read_csv(path)
        np.testing.assert_array_equal(c.row,b.row);np.testing.assert_array_equal(c.fold,b.fold);np.testing.assert_allclose(c.Y_true,b.Y_true)
        result['v16:'+family]=c.Y_pred.to_numpy(float)
        provenance.append(dict(name='v16:'+family,path=str(path),sha256=sha(path)))
    return result,provenance


def shortlist(task):
    dest=OUT/task;dest.mkdir(parents=True,exist_ok=True);locked=dest/'shortlist.json'
    if locked.exists():assert read(locked)['protocol_sha256']==sha(PFILE);return read(locked)
    b=anchor(task);pool,provenance=historical_candidates(task,b);metric=get_metric(task)
    excluded=read(Path('admet_model_outputs_v15')/task/'shortlist.json')['selected']
    y=b.Y_true.to_numpy();base=b.Y_pred.to_numpy();fold=b.fold.to_numpy();ranked=[]
    for name,c in pool.items():
        if name in excluded:continue
        modes=['raw'] if metric=='MAE' else ['raw','rank']
        for mode in modes:
            gains=[score(y[fold==f],base[fold==f],combine(base[fold==f],[c[fold==f]],[.5,.5],mode,metric),metric) for f in range(3)]
            ranked.append(dict(name=name,mode=mode,mean_gain=float(np.mean(gains))))
    ranked.sort(key=lambda r:(-r['mean_gain'],r['name'],r['mode']));historical=ranked[0]['name']
    selected=['v16:'+f for f in protocol()['new_families']]+[historical]
    result=dict(task=task,selected=selected,historical=historical,ranked=ranked,provenance=provenance,
        protocol_sha256=sha(PFILE),selection_data='Reused OOF, before new fits and official-valid evaluation')
    save(locked,result);b.to_csv(dest/'baseline_oof.csv',index=False);emit(stage='v16_shortlist',task=task,selected=selected)
    return result


def train(task):
    d=load_task(task);n=len(d['frames']['train']);fold=partition(task,d);y=d['frames']['train'].Y.to_numpy(float)
    for family in protocol()['new_families']:
        path=OUT/task/(family+'_oof.csv')
        if path.exists():continue
        q=np.full(n,np.nan)
        for f in range(3):
            it=np.flatnonzero(fold!=f);iv=np.flatnonzero(fold==f)
            assert not set(d['groups'][it])&set(d['groups'][iv])
            assert not set(d['canonical'][it])&set(d['canonical'][iv])
            q[iv]=materialize(task,d,it,iv,['v16:'+family],f'fold{f}')[0]
        assert np.isfinite(q).all()
        pd.DataFrame(dict(row=np.arange(n),fold=fold,Y_true=y,Y_pred=q)).to_csv(path,index=False)
        emit(stage='v16_oof',task=task,family=family)


def repeat_anchor(task,d,it,iv):
    p=Path('admet_model_outputs_v15')/task/'repeat_comparison.csv'
    if p.exists():
        frame=pd.read_csv(p);np.testing.assert_array_equal(frame.row,iv);np.testing.assert_allclose(frame.Y_true,d['frames']['train'].Y.to_numpy()[iv]);return frame.blend.to_numpy()
    from admet_v12_runner import repeat_baseline
    route=read(Path('admet_release_v14')/task/'selection.json')
    if route['version']==11:return repeat_baseline(task,d,it,iv)
    p=Path(f'admet_model_outputs_v{route["version"]}')/task/'repeat_comparison.csv'
    frame=pd.read_csv(p);np.testing.assert_array_equal(frame.row,iv);np.testing.assert_allclose(frame.Y_true,d['frames']['train'].Y.to_numpy()[iv]);return frame.blend.to_numpy()
''' +s[b:]
a=s.index("        from admet_v12_runner import repeat_baseline",s.index('def evaluate'));b=s.index('        active=',a)
s=s[:a]+"        br=repeat_anchor(task,d,it,iv)\n"+s[b:]
s=s.replace("choices=['shortlist','evaluate']","choices=['shortlist','train','evaluate']").replace("(shortlist if args.stage=='shortlist' else evaluate)(t)","globals()[args.stage](t)")
write('admet_v16_runner.py',s)
s=Path('admet_v15_release.py').read_text(encoding='utf-8').replace('v15','v16').replace('v14','v15').replace('49.419624','49.750495')
s=s.replace("UPLOAD=Path('admet_submissions/v16')/OLD.name","UPLOAD=Path('admet_submissions/v16')/(OLD.stem.rsplit('-',1)[0]+'-20260926.csv')")
s=s.replace('import OUT,ROOT,read,protocol,names,anchor,materialize,candidates','import OUT,ROOT,read,protocol,names,anchor,materialize,candidates,repeat_anchor')
s=s.replace("bundle=joblib.load(path);np.testing.assert_array_equal(bundle['fit_rows'],expected_fit)","bundle=joblib.load(path);np.testing.assert_array_equal(bundle['fit_rows'],expected_fit)\n        model=bundle['model']\n        if 'inner_fit_rows' in model:\n            a=model['inner_fit_rows'];b=model['inner_hold_rows'];assert set(a)|set(b)==set(expected_fit) and not set(a)&set(b)\n            assert not set(d['groups'][a])&set(d['groups'][b])")
needle="        cv=replay(t,'valid',d,np.arange(n),np.arange(n,n+nv))"
s=s.replace(needle,"""        bb=anchor(t)
        for f in range(3):
            it=np.flatnonzero(bb.fold!=f);iv=np.flatnonzero(bb.fold==f)
            values=replay(t,f'fold{f}',d,it,iv)
            for family in protocol()['new_families']:
                stored=pd.read_csv(OUT/t/(family+'_oof.csv'));np.testing.assert_allclose(stored.Y_pred.to_numpy()[iv],values['v16:'+family],atol=5e-6,rtol=5e-6)
        short=read(OUT/t/'shortlist.json')
        for source in short['provenance']:assert sha(source['path'])==source['sha256']
"""+needle)
s=s.replace("expected=combine(savedr.baseline,[rep.get", "np.testing.assert_allclose(savedr.baseline,repeat_anchor(t,d,it,iv),atol=1e-10,rtol=1e-10)\n            expected=combine(savedr.baseline,[rep.get")
write('admet_v16_release.py',s)
s=Path('admet_v15_coordinate.py').read_text(encoding='utf-8').replace('v15','v16')
s=s.replace("    run_workers('evaluate'", "    run_workers('train','admet_v16_runner.py',parts)\n    run_workers('evaluate'")
write('admet_v16_coordinate.py',s)
s=Path('admet_v15_freeze.py').read_text(encoding='utf-8').replace('v15','v16').replace("[*range(6,14),15]","[*range(6,14),15,16]")
write('admet_v16_freeze.py',s)
