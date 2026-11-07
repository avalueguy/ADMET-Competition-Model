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
CONFIGS = (dict(name='simplex_reg01', penalty=.01, rank_penalty=0.),
           dict(name='simplex_reg05', penalty=.05, rank_penalty=0.),
           dict(name='simplex_pair03', penalty=.01, rank_penalty=.3),
           dict(name='simplex_pair10', penalty=.01, rank_penalty=1.))
WEIGHTS = (.1, .25, .5, .75, 1.)
SETTINGS = dict(inner_group_folds=3, C=30., epsilon=0., tolerance=1e-5,
                maximum_iterations=1000000, maximum_pairs=4096,
                minimum_label_gap_IQR=.1, rank_margin=.05,
                simplex_prior=(.4, .4, .2), maximum_bias_IQR=.2,
                bias_penalty=.03, pair_seed=3811105)
sha, save, emit = histogram.sha, histogram.save, histogram.emit


def signature():
    code = ''.join(sha(p) for p in (__file__, histogram.__file__, histogram.base.__file__,
                                  histogram.base.shared.__file__, ROOT/'data_manifest.json'))
    return hashlib.sha256((code+json.dumps([CONFIGS, WEIGHTS, SETTINGS], sort_keys=True)).encode()).hexdigest()


def setup():
    histogram.setup()


def load():
    return histogram.load()


def ctrl(t, p):
    return histogram.ctrl(t, p)


def design(d, t, p):
    return histogram.design(d, t, p)


def configs(t):
    return CONFIGS if t in TASKS[5:] else CONFIGS[:2]


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


def pair_sample(y, groups):
    _, _, scale = histogram.base.shared.target(y)
    rng = np.random.default_rng(SETTINGS['pair_seed'])
    pairs = rng.integers(len(y), size=(SETTINGS['maximum_pairs']*8, 2))
    pairs.sort(axis=1)
    pairs = np.unique(pairs, axis=0)
    delta = y[pairs[:,1]]-y[pairs[:,0]]
    ok = ((groups[pairs[:,0]] != groups[pairs[:,1]]) &
          (np.abs(delta) > SETTINGS['minimum_label_gap_IQR']*scale))
    pairs = pairs[ok]
    rng.shuffle(pairs)
    return pairs[:SETTINGS['maximum_pairs']]


def linear_problem(predictions, y, groups, c):
    yy, center, scale = histogram.base.shared.target(y)
    x = (predictions-center)/scale
    n, e = x.shape
    pairs = pair_sample(y, groups) if c['rank_penalty'] else np.empty((0,2), dtype=int)
    r = len(pairs)
    # Variables: simplex weights, bias, absolute errors, abs prior shifts, abs bias, pair hinges.
    bias, err, shift = e, e+1, e+1+n
    abias, hinge = shift+e, shift+e+1
    total = hinge+r
    cost = np.zeros(total)
    cost[err:shift], cost[shift:abias], cost[abias] = 1/n, c['penalty'], SETTINGS['bias_penalty']
    if r:
        cost[hinge:] = c['rank_penalty']/r
    rows = 2*n+2*e+2+r
    A, rhs = sparse.lil_matrix((rows,total)), np.zeros(rows)
    A[:n,:e], A[:n,bias], A[:n,err:shift], rhs[:n] = x, 1, -sparse.eye(n), yy
    A[n:2*n,:e], A[n:2*n,bias], A[n:2*n,err:shift], rhs[n:2*n] = -x, -1, -sparse.eye(n), -yy
    prior = np.array(SETTINGS['simplex_prior'])
    for j in range(e):
        A[2*n+j,j], A[2*n+j,shift+j], rhs[2*n+j] = 1, -1, prior[j]
        A[2*n+e+j,j], A[2*n+e+j,shift+j], rhs[2*n+e+j] = -1, -1, -prior[j]
    row = 2*n+2*e
    A[row,bias], A[row,abias], A[row+1,bias], A[row+1,abias] = 1, -1, -1, -1
    if r:
        signs = np.sign(y[pairs[:,1]]-y[pairs[:,0]])
        diff = signs[:,None]*(x[pairs[:,1]]-x[pairs[:,0]])
        A[row+2:,:e], A[row+2:,hinge:], rhs[row+2:] = -diff, -sparse.eye(r), -SETTINGS['rank_margin']
    equality = sparse.lil_matrix((1,total)); equality[0,:e] = 1
    bounds = [(0,1)]*e+[(-SETTINGS['maximum_bias_IQR'],SETTINGS['maximum_bias_IQR'])]+[(0,None)]*(total-e-1)
    return cost, A.tocsr(), rhs, equality.tocsr(), bounds, pairs, scale


def fit_simplex(oof, y, groups, c):
    cost, A, rhs, E, bounds, pairs, scale = linear_problem(oof,y,groups,c)
    result = linprog(cost, A_ub=A, b_ub=rhs, A_eq=E, b_eq=[1.], bounds=bounds, method='highs')
    assert result.success, result.message
    state = dict(weights=result.x[:3], bias=float(result.x[3]*scale), solution=result.x,
                 objective=float(result.fun), pairs=pairs, scale=scale,
                 inequality_dual=result.ineqlin.marginals, equality_dual=result.eqlin.marginals,
                 lower_dual=result.lower.marginals, upper_dual=result.upper.marginals)
    check_certificate(oof,y,groups,c,state)
    return state


def check_certificate(oof, y, groups, c, s):
    cost,A,rhs,E,bounds,pairs,scale = linear_problem(oof,y,groups,c)
    np.testing.assert_array_equal(pairs,s['pairs'])
    z = s['solution']
    primal = rhs-A@z
    assert primal.min() > -2e-6 and np.max(np.abs(E@z-1)) < 1e-7
    assert abs(cost@z-s['objective']) < 1e-7 and abs(scale-s['scale']) < 1e-10
    lower = np.array([b[0] for b in bounds]); upper = np.array([np.inf if b[1] is None else b[1] for b in bounds])
    assert np.all(z>=lower-1e-7) and np.all(z<=upper+1e-7)
    stationarity = cost-A.T@s['inequality_dual']-E.T@s['equality_dual']-s['lower_dual']-s['upper_dual']
    assert np.max(np.abs(stationarity)) < 2e-6
    assert np.max(np.abs(primal*s['inequality_dual'])) < 2e-6
    assert np.max(np.abs((z-lower)*s['lower_dual'])) < 2e-6
    finite = np.isfinite(upper)
    assert np.max(np.abs((upper[finite]-z[finite])*s['upper_dual'][finite]),initial=0) < 2e-6
    assert s['inequality_dual'].max(initial=0) < 1e-7 and s['lower_dual'].min(initial=0)>-1e-7 and s['upper_dual'].max(initial=0)<1e-7
    return dict(primal_feasible=True, dual_stationarity=True, complementary_slackness=True, no_optimization_performed=True)


def clip(t, p):
    return np.clip(p,0,100) if t=='ppbr_az' else np.maximum(p,0) if t in TASKS[5:] else p


def predict(d,t,p,c):
    setup()
    path = OUT/'tasks'/t/p/c['name']/'predictions.npz'
    if path.exists():
        z=np.load(path); assert str(z['signature'])==signature(); return z['p']
    a=design(d,t,p); o=grouped_oof(d,t,p)
    state=fit_simplex(o['oof'],d['y'][a['fit']],d['rg'][a['fit']],c)
    prediction=clip(t,o['full']@state['weights']+state['bias'])
    state.update(signature=signature(),config=c,fit=a['fit'],query=a['query'])
    path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(state,path.with_name('state.joblib'),compress=3)
    np.savez_compressed(path,signature=signature(),fit=a['fit'],query=a['query'],p=prediction)
    save(path.with_name('metadata.json'),dict(signature=signature(),model_sha256=sha(path.with_name('state.joblib')),
         prediction_sha256=sha(path),certificate_checked=True,weights=state['weights'].tolist(),bias=state['bias']))
    emit('stack_done',task=t,part=p,config=c['name'],weights=state['weights'].tolist())
    return prediction


def replay(d,t,p,c):
    a=design(d,t,p)
    s=joblib.load(OUT/'tasks'/t/p/c['name']/'state.joblib')
    assert s['signature']==signature()
    for key in ('fit','query'):np.testing.assert_array_equal(s[key],a[key])
    full=replay_base(d,a,t,p,'full')
    return clip(t,full@s['weights']+s['bias'])


def freeze():
    save(OUT/'protocol.json',dict(signature=signature(),tasks=TASKS,configs=CONFIGS,settings=SETTINGS,weights=WEIGHTS,
         all_fitting_remote=True,external_data=False,external_weights=False,test_labels_used=False,
         architecture='Three converged epsilon0 SVRs: sqrt-histogram native median, Tanimoto native median, sqrt-histogram Gaussian rank mapped by fit-label empirical quantiles.',
         fit='Three inner GroupKFold OOF predictions per outer partition, each rebuilding fit-only feature scaling. Convex MAE simplex and, for four ranking tasks, cross-scaffold pair hinge. Primal/dual LP certificates retained.',
         difference_from_v372='New sqrt-count geometry and rank-normal expert, joint MAE/pairwise objective, extends all9; no tree gating or claim that stacking itself is new.',
         budget='18 development outer partitions x4 inner/full sets x3 converged SVRs =216 base fits. All completed before unique fold0/1 selection. No early rejection of families.',
         development_fits=216,development_prediction_packages=52,
         selection='Unique maxmin fold0/1 config and outer weight, frozen before fold2/valid; no confirmation fallback.',
         deployment='Same native linear output for MAE, Spearman and CSV; four positive MAE gains and four positive rank gains for ranking tasks.',
         limitations='Original historical ensemble OOF partly proxy and rank MAE reference calibrated; public validation reused. Pair loss is a surrogate, not guaranteed Spearman improvement. No five-percent or official score promise.'))
