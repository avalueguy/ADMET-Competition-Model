"""Independent v74 model replay, row identities, freeze order and byte checks."""
from pathlib import Path
import sys,json,csv,platform
import numpy as np
import torch
from catboost import CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v74_specialists as v
from finalize_v74 import reload_neural
from select_v74 import VARIANTS
from verify_submission import verify,read
from repack_upload_template import field_spans

def verify_tree(d,folder,k):
    z=np.load(folder/'predictions.npz');fit,q=z['fit'],z['query'];assert str(z['signature'])==v.signature();b.assert_excluded(d,fit,q)
    state=torch.load(folder/'state.pt',map_location='cpu',weights_only=False);pre=b.preprocess(d,fit)
    for key,value in pre.items():np.testing.assert_array_equal(value,state['pre'][key])
    tf=b.target(d['y'][fit],'MAE');assert tf==state['target'];x=b.transform(d['x'][d['mi'][q]],pre)
    net=CatBoostRegressor();net.load_model(str(folder/'model.cbm'));p=net.predict(x)*tf['scale']+tf['center']
    if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
    np.testing.assert_allclose(p,z['p'],atol=1e-8,rtol=1e-8)
    return dict(path=(folder/'model.cbm').relative_to(ROOT).as_posix(),sha256=b.sha(folder/'model.cbm'),max_error=float(np.max(abs(p-z['p']))))

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    d=b.build();g=v.graphs(d);checks=[]
    release=json.loads((v.OUT/'release.json').read_text(encoding='utf-8'));assert release['signature']==v.signature()
    frozen=json.loads((v.OUT/'frozen_selection.json').read_text(encoding='utf-8'));confirmation=json.loads((v.OUT/'confirmation.json').read_text(encoding='utf-8'))
    assert b.sha(v.OUT/'frozen_selection.json')==confirmation['frozen_selection_sha256']
    assert b.sha(ROOT/'tools/select_v74.py')==frozen['selection_source_sha256']
    assert b.sha(ROOT/'tools/run_v74_confirmation.py')==frozen['confirmation_runner_sha256']
    accepted=confirmation['accepted'];assert sorted(accepted)==sorted(release['selected'])
    for graph in g['graphs']:
        if graph is None:continue
        a,e,s,t,r=graph
        np.testing.assert_array_equal(s[r],t);np.testing.assert_array_equal(r[r],np.arange(len(r)))
        assert np.isfinite(a).all() and np.isfinite(e).all()
    for task in v.P['tasks']:
        k=d['tasks'].index(task)
        for file in sorted((v.OUT/task).glob('*/outer*/predictions.npz'))+sorted((v.OUT/task).glob('*/inner/predictions.npz'))+sorted((v.OUT/task/'final').glob('*/predictions.npz')):
            if file.parent.name=='tree':continue
            z=np.load(file);part=file.parent.parent.name
            fit,q=v.rows(d,k,part)
            if file.parent.name=='inner':fit,q=v.inner(d,fit,k)
            np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q)
            _,rec=reload_neural(d,g,k,file.parent,True);checks.extend(rec)
        for file in sorted(set((v.OUT/task).glob('*/tree*/predictions.npz'))):
            z=np.load(file);part=file.parent.parent.name
            fit,q=v.rows(d,k,part)
            if file.parent.name=='tree_inner':fit,q=v.inner(d,fit,k)
            np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q)
            checks.append(verify_tree(d,file.parent,k))
        b.emit('v74_verify',task=task,models=len(checks))
    # Recalculate the frozen selection solely from folds 0 and 1, no confirmation.
    from select_v74 import expert
    cp=np.load(ROOT/'recovered_history/baseline_cv_proxy.npy')
    for r in frozen['tasks']:
        for c in r['grid']:
            gains=[];proxy=[]
            for part in ('fold0','fold1'):
                p,tree,q=expert(r['task'],part,c);gains.append(b.gain('MAE',d['y'][q],tree,p));proxy.append(b.gain('MAE',d['y'][q],cp[q],.5*cp[q]+.5*p))
            np.testing.assert_allclose(gains,c['same_rows_tree_gains'],atol=1e-12);np.testing.assert_allclose(proxy,c['proxy_blend_gains'],atol=1e-12)
        eligible=[c for c in r['grid'] if min(c['same_rows_tree_gains'])>0 and np.mean(c['proxy_blend_gains'])>0 and min(c['proxy_blend_gains'])>-.005]
        selected=max(eligible,key=lambda c:c['objective']) if eligible else None
        assert (selected is None)==(r['selection'] is None)
        if selected:assert dict(**selected,anchor_weight=.5)==r['selection']
    vp=np.load(ROOT/'recovered_history/baseline_valid.npy')
    for r in confirmation['tasks']:
        original=next(a for a in frozen['tasks'] if a['task']==r['task'])['selection']
        if original is None:assert not r['accepted'];continue
        assert r['selection']==original
        gains=[];proxy=[]
        for part in ('fold2','valid'):
            p,tree,q=expert(r['task'],part,original);old=vp[q] if part=='valid' else cp[q]
            gains.append(b.gain('MAE',d['y'][q],tree,p));proxy.append(b.gain('MAE',d['y'][q],old,.5*old+.5*p))
        assert r['accepted']==bool(min(gains)>0 and np.mean(gains)>.002 and min(proxy)>0)
    file=ROOT/release['file'];template=ROOT/'admet_official/submission_template.csv';base=next((ROOT/'admet_submissions/v73_v72').glob('*.csv'))
    assert b.sha(file)==release['sha256'];assert b.sha(base)==v.P['anchor_sha256']
    audit=verify(file,template);new,old=read(file),read(base);changed={}
    for task in d['tasks']:
        a=[r for r in new[1:] if r[0]==task];o=[r for r in old[1:] if r[0]==task]
        if task not in accepted:assert a==o;continue
        c=release['selected'][task]['choice'];ps=[]
        for seed in v.P['final_seeds']:ps.append(np.load(v.OUT/task/'final'/str(seed)/'predictions.npz')['p'])
        p=np.mean(ps,axis=0)[c['heads']].mean(0)
        if c['tree_weight']:p=(1-c['tree_weight'])*p+c['tree_weight']*np.load(v.OUT/task/'final/tree/predictions.npz')['p']
        expected=(1-c['anchor_weight'])*np.array([float(r[4]) for r in o])+c['anchor_weight']*p
        np.testing.assert_array_equal(expected,np.array([float(r[4]) for r in a]));changed[task]=sum(r[4]!=s[4] for r,s in zip(a,o))
    raw=file.read_bytes();ref=template.read_bytes();s=field_spans(raw);t=field_spans(ref);chunks=[];cursor=0
    for i in range(1,len(new)):
        lo,hi=s[i][4];a,z=t[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    result=dict(status='PASS',signature=v.signature(),models=len(checks),final_models=sum('/final/' in r['path'] for r in checks),checks=checks,independent_process=True,
        fit_query_identity=True,inner_scaffold_exclusion=True,outer_scaffold_exclusion=True,model_reload=True,batch_invariance=True,
        frozen_selection_recomputed=True,fusion_recomputed=True,nonprediction_bytes_exact=True,changed_tasks=changed,unchanged_tasks=22-len(changed),test_labels_used=False,python=platform.python_version())
    b.save(v.OUT/'model_verification.json',result);release.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification_sha256=b.sha(v.OUT/'model_verification.json'));b.save(v.OUT/'release.json',release)
    audit.update(independent_process_verification='PASS',nonprediction_bytes_exact=True,fusion_recomputed=True,changed_tasks=changed);b.save(file.with_suffix('.verification.json'),audit)
    print(json.dumps({key:value for key,value in result.items() if key!='checks'}),flush=True)

if __name__=='__main__':main()
