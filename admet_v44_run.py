"""Remote cross-scaffold molecular difference learning with experimental anchors."""
import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import torch
from xgboost import XGBRegressor
import admet_v24_models as b
import admet_v33_run as n
import admet_v43_run as previous

ROOT = n.ROOT
OUT = ROOT / 'admet_model_outputs_v44'
P = json.loads((ROOT / 'admet_v44_protocol.json').read_text())


def signature():
    return hashlib.sha256(json.dumps(dict(base=b.signature(), previous=previous.signature(),
        files={name: b.sha(ROOT/name) for name in ['admet_v44_run.py', 'admet_v44_protocol.json']}), sort_keys=True).encode()).hexdigest()


def baseline(d):
    valid = np.load(previous.OUT / 'baseline_valid.npy').copy()
    cv = np.load(previous.OUT / 'baseline_cv_proxy.npy').copy()
    folds = np.load(b.OUT / 'folds.npy')
    report = json.loads((previous.OUT / 'analysis.json').read_text())
    assert report['signature'] == previous.signature()
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        k = d['tasks'].index(r['task'])
        vi = np.flatnonzero((d['tid']==k)&(d['split']==1))
        tr = np.flatnonzero((d['tid']==k)&(d['split']==0))
        z = np.load(previous.OUT/r['task']/'valid/predictions.npz')
        assert str(z['signature']) == previous.signature()
        p = previous.variants(z['p'], r['metric'])[s['variant']]
        new = b.blend(r['metric'], valid[vi], p, s['weight'])
        np.testing.assert_allclose(b.delta(r['metric'], d['y'][vi], valid[vi], new), s['valid_gain'], atol=1e-10, rtol=1e-8)
        valid[vi] = new
        cp = np.load(previous.OUT/r['task']/'cv_predictions.npy')[s['variant']]
        for fold in range(3):
            ix = tr[folds[tr]==fold]
            cv[ix] = b.blend(r['metric'], cv[ix], cp[ix], s['weight'])
    np.save(OUT/'baseline_valid.npy', valid)
    np.save(OUT/'baseline_cv_proxy.npy', cv)
    return valid, cv


def neighbors(d, queries, refs, count):
    count = min(count, len(refs))
    indexes, scores = [], []
    state = n.retrieval_state(d)
    groups = state['group_id']
    for start in range(0,len(queries),256):
        q = queries[start:start+256]
        sim = n.similarity(d,q,refs)
        prohibited = groups[q,None] == groups[refs][None]
        sim.masked_fill_(torch.tensor(prohibited,device='cuda'),-1.)
        value, ix = sim.topk(count, dim=1)
        indexes.append(ix.cpu().numpy())
        scores.append(value.cpu().numpy())
    ix, score = np.concatenate(indexes), np.concatenate(scores)
    assert np.all(score>=0), 'Not enough permitted cross-scaffold anchors'
    return ix, score


def design(d, rows, query):
    refs, labels = n.aggregate(d, rows)
    metric = d['metrics'][int(d['tid'][rows[0]])]
    target = b.Target(labels, metric)
    raw = np.concatenate([d['desc'], d['emb']],1)
    pre = b.Preprocessor().fit(raw[refs])
    dense = pre.transform(raw)
    fp = np.concatenate([d['fp2'],d['fcfp']],1)
    chosen = np.argsort(np.var(fp[refs],axis=0),kind='stable')[-P['pair_fingerprint_columns']:]
    point = np.concatenate([dense,fp[:,chosen]],1)
    return dict(refs=refs, labels=labels, y=target.encode(labels), target=target, pre=pre, dense=dense,
        fp=fp[:,chosen], chosen=chosen, point=point, desc_dim=d['desc'].shape[1])


def pair_matrix(x, q, r, score):
    a, c = x['dense'][q], x['dense'][r]
    dim = x['desc_dim']
    diff = a-c
    fp_a, fp_c = x['fp'][q], x['fp'][r]
    return np.concatenate([a[:,:dim], c[:,:dim], diff, np.abs(diff),
        fp_a-fp_c, np.minimum(fp_a,fp_c), score[:,None]],1).astype(np.float32)


def make_model(seed, depth, trees, pair=False):
    return XGBRegressor(n_estimators=trees, max_depth=depth, learning_rate=P['learning_rate'],
        min_child_weight=32 if pair else 4, subsample=.8, colsample_bytree=.8,
        reg_lambda=30. if pair else 18., reg_alpha=.10, objective='reg:absoluteerror',
        tree_method='hist', device='cuda', max_bin=128, n_jobs=3, random_state=seed)


def weighted_median(values, weights):
    order = np.argsort(values,axis=1,kind='stable')
    vv = np.take_along_axis(values,order,axis=1)
    ww = np.take_along_axis(weights,order,axis=1)
    ix = np.argmax(np.cumsum(ww,axis=1) >= .5*np.sum(ww,axis=1)[:,None],axis=1)
    return vv[np.arange(len(vv)),ix]


def reconstruct(d, x, query, indexes, scores, model):
    refs = x['refs']
    values = np.empty(indexes.shape,np.float64)
    for start in range(0,len(query),64):
        ids = indexes[start:start+64]
        q = np.repeat(d['mi'][query[start:start+64]],ids.shape[1])
        r = refs[ids.flatten()]
        sim = scores[start:start+64].flatten()
        forward = model.predict(pair_matrix(x,q,r,sim))
        backward = model.predict(pair_matrix(x,r,q,sim))
        diff = .5*(forward-backward)
        values[start:start+64] = x['y'][ids]+diff.reshape(ids.shape)
    # A floor avoids all-zero weights for distant chemistry.
    weight = np.maximum(scores,.05)**4
    means = []
    for count in [4,16]:
        width = min(count,indexes.shape[1])
        means.append((values[:,:width]*weight[:,:width]).sum(1)/weight[:,:width].sum(1))
    means.append(weighted_median(values,weight))
    pred = np.stack(means)
    return x['target'].decode(pred) if x['target'].m == 'MAE' else pred


def fit(d,k,rows,query,dest):
    dest.mkdir(parents=True,exist_ok=True)
    sig = signature()
    path = dest/'predictions.npz'
    if path.exists():
        z = np.load(path)
        assert str(z['signature']) == sig
        np.testing.assert_array_equal(z['fit'],rows)
        np.testing.assert_array_equal(z['query'],query)
        return z['p'].copy()
    assert not set(rows)&set(query) and np.all(d['split'][rows]<2) and np.all(d['tid'][rows]==k)
    if np.any(d['split'][query]<2):
        assert not set(d['rg'][rows])&set(d['rg'][query])
        assert not set(d['canonical'][d['mi'][rows]])&set(d['canonical'][d['mi'][query]])
    t0 = time.time()
    x = design(d,rows,query)
    indexes, scores = neighbors(d,x['refs'],x['refs'],P['training_neighbors'])
    q = np.repeat(x['refs'],indexes.shape[1])
    r = x['refs'][indexes.flatten()]
    sim = scores.flatten()
    assert not np.any(d['groups'][q] == d['groups'][r])
    y = (np.repeat(x['y'],indexes.shape[1])-x['y'][indexes.flatten()]).astype(np.float32)
    forward = pair_matrix(x,q,r,sim)
    reverse = pair_matrix(x,r,q,sim)
    pairs = np.concatenate([forward,reverse])
    targets = np.concatenate([y,-y])
    del forward,reverse
    iq,sq = neighbors(d,d['mi'][query],x['refs'],P['query_neighbors'])
    parts = []
    for seed,depth,trees in zip(P['seeds'],P['depths'],P['trees']):
        point = make_model(seed,depth,trees)
        point.fit(x['point'][x['refs']],x['y'])
        point.save_model(dest/f'point_{seed}.ubj')
        pp = point.predict(x['point'][d['mi'][query]])
        if d['metrics'][k] == 'MAE':
            pp = x['target'].decode(pp)
        restored = make_model(seed,depth,trees)
        restored.load_model(dest/f'point_{seed}.ubj')
        restored.set_params(device='cuda',n_jobs=3)
        zz = restored.predict(x['point'][d['mi'][query]])
        if d['metrics'][k] == 'MAE':
            zz = x['target'].decode(zz)
        np.testing.assert_allclose(pp,zz,rtol=2e-6,atol=2e-6)
        del point,restored
        pair = make_model(seed,depth,trees,True)
        pair.fit(pairs,targets)
        pair.save_model(dest/f'pair_{seed}.ubj')
        pred = reconstruct(d,x,query,iq,sq,pair)
        restored = make_model(seed,depth,trees,True)
        restored.load_model(dest/f'pair_{seed}.ubj')
        restored.set_params(device='cuda',n_jobs=3)
        np.testing.assert_allclose(pred,reconstruct(d,x,query,iq,sq,restored),rtol=2e-6,atol=2e-6)
        parts.append(np.concatenate([pp[None],pred]))
        del pair,restored
        b.emit('v44_seed',task=d['tasks'][k],partition=dest.name,seed=seed,seconds=round(time.time()-t0))
    p = np.mean(parts,axis=0)
    assert np.isfinite(p).all()
    np.savez_compressed(path,p=p,fit=rows,query=query,signature=sig)
    torch.save(dict(target=x['target'],pre=x['pre'],chosen=x['chosen'],refs=x['refs'],labels=x['labels'],
        signature=sig),dest/'preprocessing.pt')
    b.save(dest/'metadata.json',dict(signature=sig,task=d['tasks'][k],fit_rows=len(rows),query_rows=len(query),
        canonical_anchors=len(x['refs']),training_directed_pairs=len(targets),paired_rows_correlated=True,
        all_training_pairs_cross_scaffold=True,query_anchors_fit_only=True,query_labels_used=False,
        pair_prediction_antisymmetric=True,query_max_similarity_quantiles=np.quantile(sq[:,0],[0,.1,.5,.9,1]).tolist(),
        source_reference_rows=rows.tolist(),preprocessor_fit_only=True,test_labels_used=False,
        model_reload_passed=True,seconds=time.time()-t0,local_gpu_training=False))
    del pairs,x
    torch.cuda.empty_cache()
    return p


def variants(p,metric):
    v = p if metric == 'MAE' else np.stack([b.rank(x) for x in p])
    return np.concatenate([p,np.mean(v[1:],axis=0)[None],(.5*v[0]+.5*np.mean(v[1:],axis=0))[None]])


def screen(d):
    valid,_ = baseline(d)
    records = []
    for task in P['tasks']:
        k = d['tasks'].index(task)
        tr = np.flatnonzero((d['tid']==k)&(d['split']==0))
        vi = np.flatnonzero((d['tid']==k)&(d['split']==1))
        m = d['metrics'][k]
        vp = variants(fit(d,k,tr,vi,OUT/task/'valid'),m)
        grid = [dict(variant=v,weight=w,valid_gain=b.delta(m,d['y'][vi],valid[vi],b.blend(m,valid[vi],p,w)))
            for v,p in enumerate(vp) for w in P['weights']]
        records.append(dict(task=task,metric=m,valid_rows=len(vi),
            valid_baseline=b.metric(m,d['y'][vi],valid[vi]),candidate_metrics=[b.metric(m,d['y'][vi],p) for p in vp],
            screen_gain=max(a['valid_gain'] for a in grid),screen_grid=grid,shortlisted_variants=list(range(6))))
        b.save(OUT/'screen.json',dict(signature=signature(),status='IN_PROGRESS',tasks=records))
        b.emit('v44_screen',task=task,gain=records[-1]['screen_gain'])
    b.save(OUT/'screen.json',dict(signature=signature(),status='FROZEN',tasks=records,
        no_valid_veto=True,validation_reused=True))


def evaluate(d):
    screen_report = json.loads((OUT/'screen.json').read_text())
    assert screen_report['status']=='FROZEN' and screen_report['signature']==signature()
    base = np.load(OUT/'baseline_valid.npy')
    old = np.load(OUT/'baseline_cv_proxy.npy')
    folds = np.load(b.OUT/'folds.npy')
    cyclic = np.array([not str(a).startswith('ACYCLIC:') for a in d['groups']])
    records = []
    for r in screen_report['tasks']:
        k = d['tasks'].index(r['task'])
        m = r['metric']
        tr = np.flatnonzero((d['tid']==k)&(d['split']==0))
        vi = np.flatnonzero((d['tid']==k)&(d['split']==1))
        te = np.flatnonzero((d['tid']==k)&(d['split']==2))
        vf,tf = float(cyclic[d['mi'][vi]].mean()),float(cyclic[d['mi'][te]].mean())
        shifted = abs(vf-tf)>.5
        valid_weight = .15 if shifted else .45
        cp = np.full((6,len(d['y'])),np.nan)
        for fold in range(3):
            q = tr[folds[tr]==fold]
            rows = tr[folds[tr]!=fold]
            cp[:,q] = variants(fit(d,k,rows,q,OUT/r['task']/f'fold{fold}'),m)
        np.save(OUT/r['task']/'cv_predictions.npy',cp)
        vp = variants(np.load(OUT/r['task']/'valid/predictions.npz')['p'],m)
        grid = []
        for v in range(6):
            for w in P['weights']:
                fulls,domains,adjusted = [],[],[]
                for fold in range(3):
                    ix = tr[folds[tr]==fold]
                    new = b.blend(m,old[ix],cp[v,ix],w)
                    adj,full,dom = previous.domain_cv_gain(d,m,ix,old[ix],new,cyclic,tf)
                    fulls.append(full);domains.append(dom);adjusted.append(adj if shifted else full)
                vg = b.delta(m,d['y'][vi],base[vi],b.blend(m,base[vi],vp[v],w))
                obj = float(valid_weight*vg+(1-valid_weight)*np.mean(adjusted)-.06*np.std(adjusted))
                grid.append(dict(variant=v,weight=w,valid_gain=vg,cv_gains=fulls,cv_mean=float(np.mean(fulls)),
                    domain_cv_gains=domains,adjusted_cv_gains=adjusted,adjusted_cv_mean=float(np.mean(adjusted)),
                    objective=obj,eligible=bool(obj>.001 and np.mean(adjusted)>.0005 and sum(g>0 for g in adjusted)>=2 and vg>-.02)))
        eligible = [a for a in grid if a['eligible']]
        s = max(eligible,key=lambda a:a['objective']) if eligible else None
        records.append(dict(**r,selection=s,grid=grid,valid_cyclic_fraction=vf,test_cyclic_fraction=tf,
            domain_shift=shifted,selection_valid_weight=valid_weight))
        b.save(OUT/'analysis.json',dict(signature=signature(),status='IN_PROGRESS',tasks=records))
        b.emit('v44_selection',task=r['task'],selection=s)
    b.save(OUT/'analysis.json',dict(signature=signature(),status='EVALUATED',baseline='v43',baseline_score=P['baseline_score'],
        tasks=records,selected_tasks=[r['task'] for r in records if r['selection']],official_score_prediction=None,limitations=P['limitations']))


def final(d):
    report = json.loads((OUT/'analysis.json').read_text())
    assert report['signature']==signature() and report['status']=='EVALUATED'
    compact = {}
    for r in report['tasks']:
        if r['selection'] is None:
            continue
        k = d['tasks'].index(r['task'])
        rows = np.flatnonzero((d['tid']==k)&(d['split']<2))
        q = np.flatnonzero((d['tid']==k)&(d['split']==2))
        compact[r['task']] = fit(d,k,rows,q,OUT/r['task']/'final').tolist()
    b.save(OUT/'test_predictions_compact.json',compact)


def export(d):
    report = json.loads((OUT/'analysis.json').read_text())
    assert report['signature']==signature()
    if not report['selected_tasks']:
        b.emit('v44_no_candidate');return
    source = next((ROOT/'admet_submissions/v43').glob('*.csv'))
    assert b.sha(source)==P['baseline_sha256']
    def read(path):
        return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))
    rows = read(source)
    template = read(ROOT/'admet_official/submission_template.csv')
    compact = json.loads((OUT/'test_predictions_compact.json').read_text())
    changed = {}
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        pos = [i for i,a in enumerate(rows) if i>0 and a[0]==r['task']]
        old = np.array([float(rows[i][4]) for i in pos])
        p = variants(np.array(compact[r['task']]),r['metric'])[s['variant']]
        new = b.blend(r['metric'],old,p,s['weight'])
        assert len(new)==len(pos)
        for i,value in zip(pos,new):
            rows[i][4] = repr(float(value))
        changed[r['task']] = int(np.sum(new!=old))
    assert changed and len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,t in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==t[:4]+t[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:
            assert 0<=float(a[4])<=1
    dest = ROOT/'admet_submissions/v44'
    dest.mkdir(parents=True,exist_ok=True)
    file = dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261012.csv'
    buf = io.StringIO(newline='')
    csv.writer(buf,lineterminator='\r\n').writerows(rows)
    file.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'))
    assert read(file)==rows
    b.save(OUT/'source_manifest.json',{name:b.sha(ROOT/name) for name in
        ['admet_v44_run.py','admet_v44_protocol.json','admet_v44_remote_verify.py','admet_v44_diagnostics.py']})
    b.save(dest/'verification.json',dict(status='EXPERIMENTAL_AWAITING_PLATFORM',signature=signature(),
        baseline='v43',baseline_score=P['baseline_score'],baseline_sha256=P['baseline_sha256'],sha256=b.sha(file),
        rows=16379,tasks=22,columns=6,changed_tasks=changed,bytes=file.stat().st_size,encoding='UTF-8 BOM',
        line_endings='CRLF',template_identity_exact=True,training_host=P['training_host'],local_gpu_training=False,
        automatically_uploaded=False,platform_score=None,limitations=P['limitations']))
    b.emit('v44_export',sha256=b.sha(file),changed_tasks=changed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['screen','evaluate','final','export','all'])
    args = parser.parse_args()
    b.remote_only()
    assert ROOT==Path(P['remote_root'])
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(parents=True,exist_ok=True)
    d = b.build_data()
    if args.stage in ['screen','all']:screen(d)
    if args.stage in ['evaluate','all']:evaluate(d)
    if args.stage in ['final','all']:final(d)
    if args.stage in ['export','all']:export(d)
    b.save(OUT/f'{args.stage}_completed.json',dict(signature=signature(),completed=True))

if __name__=='__main__':
    main()
