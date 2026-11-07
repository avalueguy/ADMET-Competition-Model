"""Independent process replay of v80 nested, meta, final models and CSV bytes."""
from pathlib import Path
import sys
import json
from collections import Counter
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score as ap
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import admet_v77_ranking as data
import admet_v80_offset as v
from verify_submission import verify
from repack_upload_template import field_spans

RECORDS = []

def equal(actual, expected, tolerance=1e-11):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance, equal_nan=True)

def load(path, trees, depth, seed):
    model = CatBoostClassifier()
    model.load_model(str(path))
    params = model.get_all_params()
    assert model.tree_count_ == trees and params['depth'] == depth and params['random_seed'] == seed
    return model

def replay_residual(folder, qx, qoff):
    members = []
    for seed in v.spec()['seeds']:
        path = folder / f'model_{seed}.cbm'
        model = load(path, 300, 4, seed)
        raw = model.predict(qx, prediction_type='RawFormulaVal')
        # Model application is batch invariant; rank preprocessing uses the full cohort.
        indices = np.unique([0, len(qx)//2, len(qx)-1])
        equal(raw[indices], model.predict(qx[indices], prediction_type='RawFormulaVal'))
        members.append(raw)
        RECORDS.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=data.sha(path),
                            rows=len(qx), type='residual', status='PASS'))
    return qoff + np.mean(members, axis=0), np.stack(members)

def nested(d, k):
    out = ROOT / 'admet_model_outputs_v80_nested'
    folds = np.load(ROOT/'recovered_history/legacy_v24_folds.npy')
    aux, source, cols = v.auxiliary(d, k)
    confirmation = json.loads((out/'confirmation.json').read_text())
    assert confirmation['accepted'] and confirmation['frozen_spec'] == v.spec()
    results = []
    for rec in confirmation['records']:
        part = rec['part']; folder = out/part
        active = d['tid'] == k
        if part == 'legacy_fold2':
            fit = np.flatnonzero(active & (d['split']==0) & (folds!=2))
            query = np.flatnonzero(active & (d['split']==0) & (folds==2))
        else:
            fit = np.flatnonzero(active & (d['split']==0))
            query = np.flatnonzero(active & (d['split']==1))
        groups = np.unique(d['groups'][fit])
        order = np.random.default_rng(801020).permutation(groups)
        assignment = {str(g):j%3 for j,g in enumerate(order)}
        inner = np.array([assignment[str(g)] for g in d['groups'][fit]])
        inner_p = np.empty(len(fit))
        outer = None
        for name, tr, q in [('outer_base',fit,query)] + [
                (f'inner_base{f}',fit[inner!=f],fit[inner==f]) for f in range(3)]:
            assert not set(tr)&set(q) and np.all(d['split'][tr] == 0)
            assert not set(d['groups'][tr]) & set(d['groups'][q])
            assert not set(d['canonical'][tr]) & set(d['canonical'][q])
            cache = np.load(folder/name/'predictions.npz')
            np.testing.assert_array_equal(cache['fit'],tr)
            np.testing.assert_array_equal(cache['query'],q)
            path = folder/name/'model.cbm'
            model = load(path,1200,6,801001)
            predicted = model.predict_proba(d['views']['multiview'][d['mi'][q]])[:,1]
            equal(predicted,cache['p'])
            if name == 'outer_base':
                outer = predicted
            else:
                inner_p[inner==int(name[-1])] = predicted
            RECORDS.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=data.sha(path),
                                rows=len(q),type='nested_base',status='PASS'))
        cache = np.load(folder/'residual/predictions.npz')
        np.testing.assert_array_equal(cache['fit'],fit)
        np.testing.assert_array_equal(cache['query'],query)
        equal(cache['base_fit'],inner_p); equal(cache['base_query'],outer)
        z,qz = v.ranklog(inner_p),v.ranklog(outer)
        beta = v.calibration(d['y'][fit],z)
        qx = np.c_[aux[query],qz].astype(np.float32)
        qoff = beta[0]+beta[1]*qz
        equal(beta,cache['base_beta']); equal(qx,cache['qx']); equal(qoff,cache['qoff'])
        raw,members = replay_residual(folder/'residual',qx,qoff)
        equal(raw,cache['p']); equal(members+qoff,cache['members'])
        base,new = float(ap(d['y'][query],outer)),float(ap(d['y'][query],raw))
        equal(base,rec['baseline_ap']); equal(new,rec['candidate_ap'])
        assert new>base
        results.append(dict(part=part,baseline_ap=base,candidate_ap=new,gain=new-base))
    return results

def meta(d,k):
    out = ROOT/'admet_model_outputs_v80_meta'
    aux,_,_ = v.auxiliary(d,k)
    tr = np.flatnonzero((d['tid']==k)&(d['split']==0))
    va = np.flatnonzero((d['tid']==k)&(d['split']==1))
    folds = np.load(ROOT/'recovered_history/legacy_v24_folds.npy')
    z,vz = v.ranklog(d['proxy_cv'][tr]),v.ranklog(d['proxy_valid'][va])
    check = json.loads((out/'confirmation.json').read_text()); assert check['accepted']
    for rec in check['records']:
        part = rec['part']; folder = out/part
        if part=='valid':
            fit=tr; query=va; fz=z; qz=vz; old=d['proxy_valid'][query]
        else:
            f=int(part[-1]); mask=folds[tr]!=f; fit=tr[mask]; query=tr[~mask]
            fz=z[mask]; qz=z[~mask]; old=d['proxy_cv'][query]
        cache=np.load(folder/'predictions.npz')
        np.testing.assert_array_equal(cache['fit'],fit); np.testing.assert_array_equal(cache['query'],query)
        beta=v.calibration(d['y'][fit],fz)
        qx=np.c_[aux[query],qz].astype(np.float32); qoff=beta[0]+beta[1]*qz
        equal(beta,cache['beta']); equal(qx,cache['qx']); equal(qoff,cache['qoff'])
        raw,members=replay_residual(folder,qx,qoff)
        equal(raw,cache['p']); equal(members,cache['members'])
        equal(float(ap(d['y'][query],old)),rec['baseline_ap'])
        equal(float(ap(d['y'][query],raw)),rec['candidate_ap'])
    return check['records']

def final(d,k):
    release=json.loads((v.OUT/'release.json').read_text())
    for path,expected in release['input_hashes'].items():
        assert data.sha(ROOT/path)==expected,path
    folder=v.OUT/v.TASK; cache=np.load(folder/'predictions.npz')
    metadata=json.loads((folder/'metadata.json').read_text())
    assert metadata['state_sha256']==data.sha(folder/'predictions.npz')
    for name,expected in metadata['model_hashes'].items():
        assert data.sha(folder/name)==expected
    tr=np.flatnonzero((d['tid']==k)&(d['split']==0)); va=np.flatnonzero((d['tid']==k)&(d['split']==1))
    fit=np.r_[tr,va]; query=np.flatnonzero((d['tid']==k)&(d['split']==2))
    np.testing.assert_array_equal(cache['fit'],fit); np.testing.assert_array_equal(cache['query'],query)
    aux,source,cols=v.auxiliary(d,k,include_valid=True)
    np.testing.assert_array_equal(source,np.load(folder/'auxiliary_source_rows.npy'))
    assert k not in cols and np.all(d['tid'][source]!=k)
    original=v.read_csv(v.anchor()); file=ROOT/release['file']; actual=v.read_csv(file)
    positions=[i for i,row in enumerate(original) if i and row[0]==v.TASK]
    base=np.r_[d['proxy_cv'][tr],d['proxy_valid'][va]]
    qb=np.array([float(original[i][4]) for i in positions])
    equal(cache['base_fit'],base); equal(cache['base_query'],qb)
    z,qz=v.ranklog(base),v.ranklog(qb); beta=v.calibration(d['y'][fit],z)
    qoff=beta[0]+beta[1]*qz; qx=np.c_[aux[query],qz].astype(np.float32)
    equal(cache['beta'],beta); equal(cache['qx'],qx); equal(cache['qoff'],qoff)
    equal(cache['x'],np.c_[aux[fit],z].astype(np.float32)); equal(cache['offset'],beta[0]+beta[1]*z)
    raw,members=replay_residual(folder,qx,qoff); probability=expit(raw)
    equal(members,cache['residual_members']); equal(raw,cache['raw']); equal(probability,cache['p'])
    exported=np.array([float(actual[i][4]) for i in positions]); equal(probability,exported,1e-14)
    assert data.sha(file)==release['sha256']
    changed=Counter()
    for row,old in zip(actual[1:],original[1:]):
        assert row[:4]+row[5:]==old[:4]+old[5:]
        if row[4]!=old[4]: changed[row[0]]+=1
        if row[0]!=v.TASK: assert row==old
    assert dict(changed)=={v.TASK:2419}
    template=ROOT/'admet_official/submission_template.csv'
    audit=verify(file,template)
    # Undo only y_pred insertions and require the exact official template bytes.
    rawfile=file.read_bytes(); rawtemplate=template.read_bytes()
    spans=field_spans(rawfile); tspan=field_spans(rawtemplate); chunks=[]; cursor=0
    for i in range(1,len(actual)):
        lo,hi=spans[i][4]; a,b=tspan[i][4]
        chunks.extend((rawfile[cursor:lo],rawtemplate[a:b])); cursor=hi
    chunks.append(rawfile[cursor:]); assert b''.join(chunks)==rawtemplate
    audit.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),unchanged_tasks=21,
        independent_model_reload='PASS',model_count=3,
        test_target_rows_with_other_assays=int(np.sum(aux[query,len(cols):].sum(axis=1)>0)),
        prediction_spearman_to_anchor=float(spearmanr(qb,exported).statistic))
    file.with_suffix('.verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    release.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=audit,
                   independent_model_reload='PASS')
    data.save(v.OUT/'release.json',release)
    return audit

def main():
    d=data.load_data(); k=d['tasks'].index(v.TASK)
    assert data.sha(ROOT/'recovered_history/legacy_v24_folds.npy')==v.FOLD_SHA
    nested_results=nested(d,k); meta_results=meta(d,k); final_results=final(d,k)
    result=dict(status='PASS',models=len(RECORDS),model_records=RECORDS,
        nested=nested_results,meta_diagnostics=meta_results,submission=final_results,
        target_auxiliary_excluded=True,test_labels_used=False,external_data=False,
        rank_preprocessing='Full specified cohort; no claim of arbitrary-cohort invariance')
    data.save(v.OUT/'model_verification.json',result)
    print(json.dumps({k:x for k,x in result.items() if k!='model_records'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
