"""Remote-only grouped OOF stacks of native-median and Gaussian-rank experts.

This extends the failed v372 stack with the v380 histogram geometry and a
rank expert/joint objective. It does not recover legacy ensemble checkpoints.
"""
from pathlib import Path
import hashlib
import json
import time
import joblib
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR
import v380_histogram_median as histogram

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
TASKS = histogram.TASKS
BASES = ('sqrt_median', 'tanimoto_median', 'sqrt_gaussian_rank')
CONFIGS = tuple(dict(name=b+'_'+mode, base=b, calibration=mode, penalty=penalty)
                for b in BASES for mode,penalty in (('raw',0.),('mono02',.02),('mono10',.1)))
WEIGHTS = (.1,.25,.5,.75,1.)
SETTINGS = dict(inner_group_folds=3,C=30.,epsilon=0.,tolerance=1e-5,maximum_iterations=1000000)
import v386_calibration_core as calibration
sha, save, emit = histogram.sha, histogram.save, histogram.emit


def signature():
    code = ''.join(sha(p) for p in (__file__, histogram.__file__, histogram.base.__file__,
                                  histogram.base.shared.__file__, calibration.__file__, ROOT/'data_manifest.json'))
    return hashlib.sha256((code+json.dumps([CONFIGS, WEIGHTS, SETTINGS], sort_keys=True)).encode()).hexdigest()


def setup():
    histogram.ROOT=ROOT; histogram.base.ROOT=ROOT; histogram.base.shared.ROOT=ROOT; histogram.setup()


def load():
    histogram.ROOT=ROOT; return histogram.load()


def ctrl(t, p):
    return histogram.ctrl(t, p)


def design(d, t, p):
    return histogram.design(d, t, p)


def configs(t):
    return CONFIGS


def matrices(d, a):
    # Rebuild fit-dependent numeric scaling for every inner training fold.
    k, q = histogram.matrices(d, a)
    tan, qt, _ = histogram.base.shared.kernels(d, a)
    return dict(sqrt=k['sqrt_polynomial'], tanimoto=tan), dict(sqrt=q['sqrt_polynomial'], tanimoto=qt)


def subdesign(a, tr, ho):
    return dict(fit=a['fit'][tr], query=a['fit'][ho], x=a['x'][tr], qx=a['x'][ho], source=a['source'])


def kernel(d, a, t, p, tag, rebuild=False):
    path = OUT/'kernels'/t/p/(tag+'.npz')
    if path.exists() and not rebuild:
        z = np.load(path)
        assert str(z['signature']) == signature()
        for key in ('fit', 'query'):
            np.testing.assert_array_equal(z[key], a[key])
        return ({v:z[v+'_train'] for v in ('sqrt', 'tanimoto')},
                {v:z[v+'_query'] for v in ('sqrt', 'tanimoto')})
    k, q = matrices(d, a)
    if not rebuild:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, signature=signature(), fit=a['fit'], query=a['query'],
                            **{v+'_'+s:x for v in k for s,x in (('train', k[v]), ('query', q[v]))})
    return k, q


def native(pred, state, t):
    if state['name'] == 'sqrt_gaussian_rank':
        q = np.clip(ndtr(pred), 0, 1)
        # Nondecreasing empirical quantile map, fitted only to allowed labels.
        out = np.quantile(state['training_y'], q, method='linear')
    else:
        out = pred*state['scale']+state['center']
    if t == 'ppbr_az':
        out = np.clip(out, 0, 100)
    if t in TASKS[5:]:
        out = np.maximum(out, 0)
    return out


def fit_base(d, a, t, p, tag):
    path = OUT/'base'/t/p/tag/'state.joblib'
    if path.exists():
        s = joblib.load(path)
        assert s['signature'] == signature()
        for key in ('fit', 'query'):
            np.testing.assert_array_equal(s[key], a[key])
        return s
    started = time.monotonic()
    assert not np.intersect1d(a['fit'], a['query']).size
    assert not np.intersect1d(d['rg'][a['fit']], d['rg'][a['query']]).size or (p=='final' and tag=='full')
    y = d['y'][a['fit']]
    assert np.isfinite(y).all()
    yy, center, scale = histogram.base.shared.target(y)
    ranked = ndtri((rankdata(y, method='average')-.5)/len(y))
    k, q = kernel(d, a, t, p, tag)
    models, predictions = [], []
    for name in BASES:
        kind = 'tanimoto' if name == 'tanimoto_median' else 'sqrt'
        model = SVR(kernel='precomputed', C=SETTINGS['C'], epsilon=0,
                    tol=SETTINGS['tolerance'], max_iter=SETTINGS['maximum_iterations'], cache_size=768)
        model.fit(k[kind], ranked if name == 'sqrt_gaussian_rank' else yy)
        assert model.fit_status_ == 0, 'All optimizers must converge; preserve failures.'
        state = dict(name=name, model=model, center=center, scale=scale,
                     training_y=y.copy() if name == 'sqrt_gaussian_rank' else None,
                     iterations=int(model.n_iter_))
        models.append(state)
        predictions.append(native(model.predict(q[kind]), state, t))
    state = dict(signature=signature(), fit=a['fit'], query=a['query'],
                 models=models, predictions=np.column_stack(predictions))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, path, compress=3)
    save(path.with_name('metadata.json'), dict(signature=signature(), model_sha256=sha(path),
          iterations=[s['iterations'] for s in models], converged=True, seconds=time.monotonic()-started))
    emit('base_done', task=t, part=p, tag=tag, models=3, seconds=time.monotonic()-started)
    return state


def replay_base(d, a, t, p, tag):
    path = OUT/'base'/t/p/tag/'state.joblib'
    s = joblib.load(path)
    assert s['signature'] == signature()
    for key in ('fit', 'query'):
        np.testing.assert_array_equal(s[key], a[key])
    _, q = kernel(d, a, t, p, tag)
    return np.column_stack([native(m['model'].predict(q['tanimoto' if m['name']=='tanimoto_median' else 'sqrt']), m, t)
                            for m in s['models']])


def grouped_oof(d, t, p):
    path = OUT/'oof'/t/p/'state.joblib'
    if path.exists():
        s = joblib.load(path)
        assert s['signature'] == signature()
        return s
    a = design(d, t, p)
    oof = np.full((len(a['fit']), 3), np.nan)
    splits = []
    for i, (tr, ho) in enumerate(GroupKFold(3).split(a['fit'], groups=d['rg'][a['fit']])):
        sub = subdesign(a, tr, ho)
        oof[ho] = fit_base(d, sub, t, p, 'inner'+str(i))['predictions']
        splits.append(dict(train=tr, held=ho))
    full = fit_base(d, a, t, p, 'full')['predictions']
    assert np.isfinite(oof).all()
    state = dict(signature=signature(), fit=a['fit'], query=a['query'], splits=splits,
                 oof=oof, full=full)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, path, compress=3)
    return state


def clip(t,p):
    return np.clip(p,0,100) if t=='ppbr_az' else np.maximum(p,0) if t in TASKS[5:] else p

def predict(d,t,p,c):
    setup();path=OUT/'tasks'/t/p/c['name']/'predictions.npz'
    if path.exists():
        z=np.load(path);assert str(z['signature'])==signature();return z['p']
    a=design(d,t,p);o=grouped_oof(d,t,p);j=BASES.index(c['base']);state=None
    if c['calibration']!='raw':state=calibration.fit(o['oof'][:,j],d['y'][a['fit']],c['penalty'])
    pred=clip(t,o['full'][:,j] if state is None else calibration.apply(o['full'][:,j],state))
    s=dict(signature=signature(),config=c,fit=a['fit'],query=a['query'],calibration=state)
    path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(s,path.with_name('state.joblib'),compress=3)
    np.savez_compressed(path,signature=signature(),fit=a['fit'],query=a['query'],p=pred)
    save(path.with_name('metadata.json'),dict(signature=signature(),model_sha256=sha(path.with_name('state.joblib')),prediction_sha256=sha(path)))
    emit('calibration_done',task=t,part=p,config=c['name']);return pred

def replay(d,t,p,c):
    a=design(d,t,p);s=joblib.load(OUT/'tasks'/t/p/c['name']/'state.joblib');assert s['signature']==signature()
    for k in ('fit','query'):np.testing.assert_array_equal(s[k],a[k])
    raw=replay_base(d,a,t,p,'full')[:,BASES.index(c['base'])]
    return clip(t,raw if s['calibration'] is None else calibration.apply(raw,s['calibration']))

def freeze():
    save(OUT/'protocol.json',dict(signature=signature(),tasks=TASKS,configs=CONFIGS,weights=WEIGHTS,settings=SETTINGS,
        baseline='complete v385 associated official51.308',all_fitting_remote=True,external_data=False,external_weights=False,
        base='Three existing converged SVRs: sqrt native, Tanimoto native, sqrt Gaussian rank. Refit inner3/full with per-inner feature scaling; no historical outputs as calibration features.',
        method='Nine fit-OOF-prediction quantile knots; monotone piecewise linear MAE fit with slope bounds0.05..5, knot shift bounds0.5IQR, L1 identity shrink0.02/0.1. Outside knots fixed slope1. Raw expert controls.',
        budget='18 development outer partitions x4 inner/full x3 SVRs=216 base fits, plus18x3x2=108 calibration LPs; all before selection.',
        novelty='Extends prior affine/empirical marginal calibration and inner OOF stacking; calibrated single-expert monotone shape is the tested difference, not invention of calibration.',
        selection='Unique maxmin fold0/1 config and blend; freeze before fold2/valid; no confirmation fallback. Four MAE gains positive, and four Spearman gains positive for rank tasks.',
        limitations='Legacy proxy controls, calibrated rank MAE references, repeated public validation. Monotonicity preserves a single unbounded expert ranking, not ranking of a blend or improvement vs historical ensemble.'))
