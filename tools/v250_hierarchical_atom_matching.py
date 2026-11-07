"""Fit-only hierarchical atom-environment histogram-intersection kernels.

The threshold lift gives exact multiset intersections (minimum counts), not
count dot products/Tanimoto. No historical predictions enter the new model.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
import collections
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from rdkit import Chem, RDLogger, rdBase
from sklearn.svm import SVC, SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import admet_v67_data as parser
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT/'admet_model_outputs_v250'
TASKS = ('herg', 'clearance_hepatocyte_az', 'caco2_wang', 'ppbr_az',
         'lipophilicity_astrazeneca')
CONFIGS = (dict(C=3., levels=4),)
METHOD = ('Hierarchical atom-environment multiset matching: exact SHA256 WL '
          'refinement levels0..3, with atom chemistry and edge labels. Fitting-only '
          'environment vocabulary, count thresholds and document-frequency weights '
          'define an explicit nonnegative threshold lift. Its inner product is '
          'rarity-weighted histogram intersection, hence a PSD optimal-assignment '
          'style matching kernel, rather than count Tanimoto. Equal normalized '
          'depth kernels plus fixed physical/target-excluded context RBF channels. '
          'Exact SVC/SVR, one declared C, no prior models/prediction inputs, five '
          'eligible v145 endpoints. Original v148 selection/blends/gates unchanged.')


def sig():
    return hashlib.sha256((b.sha(__file__)+b.sha(parser.__file__)+v.sig()+
        b.sha(runner.__file__)+rdBase.rdkitVersion+
        importlib.metadata.version('scikit-learn')+importlib.metadata.version('scipy')).encode()).hexdigest()


def key(c):
    return 'hierarchical_atom_matching'


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def environments(smiles):
    mol, parse_mode = parser.parse_molecule(smiles)
    atoms = []
    for a in mol.GetAtoms():
        atoms.append(digest((a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(),
            int(a.GetIsAromatic()), a.GetDegree(), a.GetTotalNumHs(),
            a.GetProp('_CIPCode') if a.HasProp('_CIPCode') else '', int(a.IsInRing()))))
    adjacency = [[] for _ in atoms]
    for e in mol.GetBonds():
        label = (str(e.GetBondType()), int(e.GetIsAromatic()), int(e.GetIsConjugated()), int(e.GetStereo()))
        i, j = e.GetBeginAtomIdx(), e.GetEndAtomIdx()
        adjacency[i].append((label,j))
        adjacency[j].append((label,i))
    blocks = [dict(collections.Counter(atoms))]
    for _ in range(3):
        atoms = [digest((atoms[i], sorted((edge, atoms[j]) for edge,j in adj)))
                 for i,adj in enumerate(adjacency)]
        blocks.append(dict(collections.Counter(atoms)))
    assert all(sum(z.values()) == mol.GetNumAtoms() for z in blocks)
    return dict(blocks=blocks, atom_count=mol.GetNumAtoms(), parse_mode=parse_mode)


def load():
    RDLogger.DisableLog('rdApp.*')
    d = v.load()
    smiles = {}
    cursor = 0
    for task in d['tasks']:
        for part in ('train','valid','test'):
            rows = ([dict(Drug=r['smiles']) for r in d['template'] if r['task']==task]
                    if part=='test' else b.table(ROOT/'admet_competition_data/data_benchmark'/task/(part+'.csv')))
            for row in rows:
                if task in TASKS:
                    mi = int(d['mi'][cursor])
                    assert smiles.get(mi, row['Drug']) == row['Drug']
                    smiles[mi] = row['Drug']
                cursor += 1
    assert cursor == len(d['y'])
    d['matching_smiles'] = smiles
    source = digest(sorted(smiles.items()))
    feature_file = OUT/'features/atom_environments.joblib'
    audit_file = OUT/'features/audit.json'
    if audit_file.exists():
        audit = json.loads(audit_file.read_text())
        assert audit['signature']==sig() and audit['smiles_sha256']==source
        assert b.sha(feature_file)==audit['features_sha256']
        d['atom_environments'] = joblib.load(feature_file)
    else:
        assert not feature_file.exists(), 'Inspect partial feature construction before retry'
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        records = {mi:environments(sm) for mi,sm in sorted(smiles.items())}
        joblib.dump(records, feature_file, compress=3)
        b.save(audit_file, dict(status='PASS', signature=sig(), smiles_sha256=source,
            features_sha256=b.sha(feature_file), molecules=len(records), levels=4,
            original_graph_preserved=True, labels_used=False, rdkit=rdBase.rdkitVersion))
        d['atom_environments'] = records
    assert set(d['atom_environments']) == set(smiles)
    return d


def same(a, z):
    if isinstance(a, dict):
        assert set(a)==set(z)
        for name in a:
            same(a[name],z[name])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(z)
        for x,y in zip(a,z):
            same(x,y)
    else:
        np.testing.assert_array_equal(a,z)


def lift(records, n):
    # A value c is represented by indicators 1..c. Their dot product is min(c,d).
    vocabulary = sorted({name for row in records[:n] for name in row})
    maxcounts, df = {}, collections.Counter()
    for row in records[:n]:
        df.update(row.keys())
        for name,count in row.items():
            maxcounts[name]=max(maxcounts.get(name,0),count)
    df=dict(df)
    offsets, weights, cursor = {}, {}, 0
    for name in vocabulary:
        offsets[name]=cursor
        cursor += maxcounts[name]
        weights[name]=float(min(4.,1.+np.log((n+1.)/(df[name]+1.))))
    columns, values, ptr = [],[],[0]
    for row in records:
        for name, count in sorted(row.items()):
            if name not in offsets:
                continue
            count = min(count,maxcounts[name])
            columns.extend(range(offsets[name],offsets[name]+count))
            values.extend([np.sqrt(weights[name])]*count)
        ptr.append(len(columns))
    x = sparse.csr_matrix((np.array(values),np.array(columns),np.array(ptr)),shape=(len(records),cursor))
    norm = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    x = sparse.diags(1./norm.clip(1e-12))@x
    return x.tocsr(),dict(vocabulary=vocabulary, maxcounts=maxcounts,
                         document_frequency=df, offsets=offsets, weights=weights)


def rbf(raw, n):
    median = np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 0. for x in raw[:n].T])
    scale = np.array([np.std(x[np.isfinite(x)]) if np.isfinite(x).any() else 1. for x in raw[:n].T]).clip(.05)
    clean = np.where(np.isfinite(raw),raw,median)
    keep = np.flatnonzero(np.std(clean[:n],axis=0)>.01)
    z = np.clip((clean[:,keep]-median[keep])/scale[keep],-5.,5.)
    z = np.c_[z,(~np.isfinite(raw[:,keep])).astype(float)]
    norm = np.sum(z*z,axis=1)
    result = np.empty((len(raw),n))
    for lo in range(0,len(raw),256):
        xx = z[lo:lo+256]
        dist = np.maximum(norm[lo:lo+len(xx),None]+norm[None,:n]-2.*xx@z[:n].T,0.)/max(1,z.shape[1])
        result[lo:lo+len(xx)] = .5*np.exp(-dist/.5)+.5*np.exp(-dist/2.)
    return result,dict(median=median,scale=scale,keep=keep)


def design(d,k,part,c=None):
    c = c or CONFIGS[0]
    fit,query,x,qx,source = v.old.features(d,k,part,'augmented')
    rows,n = np.r_[fit,query],len(fit)
    environments_by_row = [d['atom_environments'][int(mi)]['blocks'] for mi in d['mi'][rows]]
    kernel = np.zeros((len(rows),n))
    lifts = []
    for level in range(c['levels']):
        z,state = lift([record[level] for record in environments_by_row],n)
        kernel += .75/c['levels']*(z@z[:n].T).toarray()
        lifts.append(state)
    width = d['views']['physchem'].shape[1]+d['conformer_features'].shape[1]+d['pharmacophore'].shape[1]
    raw = np.r_[x,qx].astype(float)
    physical,ps = rbf(raw[:,-width:],n)
    assay,ast = rbf(raw[:,:-width],n)
    kernel += .15*physical+.10*assay
    kernel[:n] = (kernel[:n]+kernel[:n].T)*.5
    metric,y = d['metrics'][k],d['y'][fit]
    center = float(np.median(y)) if metric=='MAE' else 0.
    scale = max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if metric=='MAE' else 1.
    target = (y-center)/scale if metric=='MAE' else r83.ranknorm(y) if metric=='Spearman' else y
    state = dict(fit=fit,query=query,source=source,lifts=lifts,physical=ps,assay=ast,
                 center=center,scale=scale,target=target)
    assert not set(fit)&set(query) and np.all(d['tid'][source]!=k)
    assert np.all(d['split'][source] < (2 if part=='final' else 1))
    if part!='final':
        assert not set(d['groups'][fit])&set(d['groups'][query])
    assert np.isfinite(kernel).all()
    return dict(x=kernel[:n],qx=kernel[n:],state=state)


def estimator(metric,c):
    return (SVC(C=c['C'],kernel='precomputed',tol=1e-5,cache_size=1024)
            if metric in ('AUROC','AUPRC') else
            SVR(C=c['C'],epsilon=.04,kernel='precomputed',tol=1e-5,cache_size=1024))


def infer(net,x,metric,state):
    p = net.decision_function(x) if metric in ('AUROC','AUPRC') else net.predict(x)
    return p*state['scale']+state['center']


def replay(d,k,part,c):
    fit,query=v.split(d,k,part)
    for mi in np.unique(d['mi'][np.r_[fit,query]]):
        # Derive all atom environments again from immutable original SMILES.
        same(environments(d['matching_smiles'][int(mi)]),d['atom_environments'][int(mi)])
    a = design(d,k,part,c)
    folder = OUT/'models'/d['tasks'][k]/part/key(c)
    same(a['state'],joblib.load(folder/'transform.joblib'))
    path = folder/'model.joblib'
    return infer(joblib.load(path),a['qx'],d['metrics'][k],a['state']),{
        'transform.joblib':b.sha(folder/'transform.joblib'),'model.joblib':b.sha(path)}


def predict(d,k,part,c):
    if d['tasks'][k] not in TASKS:
        return v.baseline(d,k,part)
    folder = OUT/'models'/d['tasks'][k]/part/key(c)
    file = folder/'predictions.npz'
    if file.exists():
        z = np.load(file)
        assert str(z['signature'])==sig()
        np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1])
        return z['p'].copy()
    started = time.monotonic()
    a = design(d,k,part,c)
    net = estimator(d['metrics'][k],c)
    net.fit(a['x'],a['state']['target'])
    assert net.fit_status_==0
    p = infer(net,a['qx'],d['metrics'][k],a['state'])
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(a['state'],folder/'transform.joblib',compress=3)
    path = folder/'model.joblib'
    joblib.dump(net,path,compress=3)
    np.testing.assert_array_equal(p,infer(joblib.load(path),a['qx'],d['metrics'][k],a['state']))
    np.savez_compressed(file,signature=sig(),fit=a['state']['fit'],query=a['state']['query'],p=p)
    print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,
                          seconds=round(time.monotonic()-started))),flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    rows = [dict(a=3,b=1),dict(a=1,c=2),dict(a=2,b=1,c=1)]
    x,state = lift(rows,2)
    raw = []
    for row in rows:
        raw.append({name:min(row.get(name,0),state['maxcounts'][name]) for name in state['vocabulary']})
    manual = np.array([[sum(state['weights'][name]*min(a[name],z[name]) for name in state['vocabulary'])
                        for z in raw] for a in raw])
    diagonal = np.sqrt(np.diag(manual))
    manual /= diagonal[:,None]*diagonal[None,:]
    np.testing.assert_allclose((x@x.T).toarray(),manual,rtol=0,atol=1e-13)
    # Query-only vocabulary and counts cannot change fitting lift geometry.
    changed = rows[:2]+[dict(a=100,query_only=100)]
    z,other = lift(changed,2)
    same(state,other)
    np.testing.assert_array_equal((x[:2]@x[:2].T).toarray(),(z[:2]@z[:2].T).toarray())
    for sm in ('CC(O)N','C[C@H](O)N','c1ncccc1CO'):
        mol=Chem.MolFromSmiles(sm)
        other=Chem.MolToSmiles(Chem.RenumberAtoms(mol,list(reversed(range(mol.GetNumAtoms())))),canonical=False)
        same(environments(sm),environments(other))
    checked=[]
    for task in ('herg','caco2_wang','clearance_hepatocyte_az'):
        k=d['tasks'].index(task)
        a=design(d,k,'fold0')
        outside=np.flatnonzero((d['tid']==k)&~np.isin(np.arange(len(d['y'])),a['state']['fit']))
        labels=d['y'][outside].copy()
        try:
            d['y'][outside]=999999.
            d.pop('_v96_assays',None)
            same(a,design(d,k,'fold0'))
        finally:
            d['y'][outside]=labels
            d.pop('_v96_assays',None)
        np.testing.assert_allclose(np.diag(a['x']),1.,rtol=0,atol=1e-12)
        assert np.linalg.eigvalsh(a['x'][:64,:64]).min()>-1e-8
        net=estimator(d['metrics'][k],CONFIGS[0])
        net.fit(a['x'][:96,:96],a['state']['target'][:96])
        assert net.fit_status_==0
        qx=a['qx'][:8,:96]
        p=infer(net,qx,d['metrics'][k],a['state'])
        np.testing.assert_array_equal(p,infer(net,qx[::-1],d['metrics'][k],a['state'])[::-1])
        with TemporaryDirectory(dir=OUT) as temp:
            path=Path(temp)/'model.joblib';joblib.dump(net,path)
            np.testing.assert_array_equal(p,infer(joblib.load(path),qx,d['metrics'][k],a['state']))
        checked.append(task)
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),
        exact_threshold_lift_reference=True,fit_vocabulary_query_invariant=True,
        atom_order_invariant=True,query_label_isolation=True,psd_reference=True,
        independent_reload_tasks=checked,prediction_row_order=True,no_query_scores_used=True))
    print('SMOKE_PASS',flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    d=load()
    if '--smoke' in sys.argv:
        smoke(d)
    else:
        assert json.loads((OUT/'smoke.json').read_text())['signature']==sig()
        runner.run(sys.modules[__name__],d)


if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback
        b.save(OUT/'failure.json',dict(status='FAILED',traceback=traceback.format_exc(),retry_permitted=False))
        raise
