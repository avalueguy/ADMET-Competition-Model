"""Remote-only square-root histogram kernels with converged absolute-loss SVR."""
from pathlib import Path
import hashlib
import json
import time
import joblib
import numpy as np
import torch
from sklearn.svm import SVR
import v377_regression_experts as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
TASKS = base.TASKS
CONFIGS = tuple(dict(name=f'{view}_C{int(c)}_{mode}', view=view, C=c, mode=mode)
                for view in ('sqrt_polynomial', 'hellinger_gaussian')
                for c in (3., 30.) for mode in ('native', 'weighted_log'))
WEIGHTS = (.1, .25, .5, .75, 1.)
SETTINGS = dict(epsilon=0., tolerance=1e-5, maximum_iterations=1000000,
                fingerprint_weight=.8, augmented_weight=.15, physical_weight=.05,
                gaussian_rates=(2., 8.), log_weight_cap=20.)
sha, save, emit = base.sha, base.save, base.emit


def signature():
    payload = sha(__file__)+sha(base.__file__)+sha(base.shared.__file__)+sha(ROOT/'data_manifest.json')
    return hashlib.sha256((payload+json.dumps([CONFIGS, WEIGHTS, SETTINGS], sort_keys=True)).encode()).hexdigest()


def setup():
    base.setup()


def load():
    base.ROOT = ROOT
    base.shared.ROOT = ROOT
    return base.load()


def ctrl(t, p):
    return base.ctrl(t, p)


def design(d, t, p):
    return base.design(d, t, p)


def configs(t):
    return CONFIGS if t in TASKS[5:] else tuple(c for c in CONFIGS if c['mode']=='native')


def histogram_map(raw):
    counts = np.rint(np.expm1(raw.astype(np.float64)))
    assert np.isfinite(counts).all() and counts.min() >= 0
    return np.sqrt(counts/counts.sum(1, keepdims=True).clip(1.)).astype(np.float32)


def matrices(d, a):
    setup()
    fi, qi = d['mi'][a['fit']], d['mi'][a['query']]
    n = len(fi)
    k = {v:np.zeros((n,n),np.float32) for v in ('sqrt_polynomial','hellinger_gaussian')}
    q = {v:np.zeros((len(qi),n),np.float32) for v in k}
    for view in range(6):
        raw = d['fingerprints'][:,view*2048:(view+1)*2048]
        ref = torch.as_tensor(histogram_map(raw[fi]), device='cuda')
        norm = ref.square().sum(1)
        for ids, dest in ((fi,k),(qi,q)):
            for lo in range(0,len(ids),256):
                x = torch.as_tensor(histogram_map(raw[ids[lo:lo+256]]), device='cuda')
                dot = (x@ref.T).clamp(0,1)
                distance = (x.square().sum(1)[:,None]+norm[None]-2*dot).clamp_min(0)
                dest['sqrt_polynomial'][lo:lo+len(x)] += (.5*dot+.5*dot.square()).cpu().numpy()/6
                dest['hellinger_gaussian'][lo:lo+len(x)] += (.5*torch.exp(-2*distance)+.5*torch.exp(-8*distance)).cpu().numpy()/6
    aug, qaug, _ = base.shared.rbf(a['x'],a['qx'])
    physical, qphysical, _ = base.shared.rbf(d['physical'][fi],d['physical'][qi])
    for name in k:
        k[name] = .8*k[name]+.15*aug+.05*physical
        q[name] = .8*q[name]+.15*qaug+.05*qphysical
        k[name] = (k[name]+k[name].T)*.5
        assert np.isfinite(k[name]).all() and np.isfinite(q[name]).all()
    return k, q


def cached(d,t,p):
    path = OUT/'kernels'/t/(p+'.npz')
    if path.exists():
        z = np.load(path)
        assert str(z['signature'])==signature()
        a=design(d,t,p)
        for key in ('fit','query'):np.testing.assert_array_equal(z[key],a[key])
        return {v:(z[v+'_train'],z[v+'_query']) for v in ('sqrt_polynomial','hellinger_gaussian')}
    a=design(d,t,p)
    k,q=matrices(d,a)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path, signature=signature(), fit=a['fit'], query=a['query'],
                        **{v+'_'+s:arr for v in k for s,arr in (('train',k[v]),('query',q[v]))})
    return {v:(k[v],q[v]) for v in k}


def target(y,c):
    if c['mode']=='weighted_log':
        assert y.min()>=0
        raw=np.log1p(y)
        weights=np.clip((1+y)/np.median(1+y),1/SETTINGS['log_weight_cap'],SETTINGS['log_weight_cap'])
    else:
        raw=y
        weights=np.ones(len(y))
    center=float(np.median(raw))
    scale=max(float(np.subtract(*np.quantile(raw,[.75,.25]))),.05)
    return (raw-center)/scale,center,scale,weights


def inverse(raw,s,t):
    p=raw*s['scale']+s['center']
    if s['config']['mode']=='weighted_log':p=np.expm1(np.clip(p,-20,20))
    if t=='ppbr_az':p=np.clip(p,0,100)
    if t in TASKS[5:]:p=np.maximum(p,0)
    return p


def predict(d,t,p,c):
    assert c in configs(t)
    setup()
    folder=OUT/'tasks'/t/p/c['name']
    file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file)
        assert str(z['signature'])==signature()
        return z['p']
    start=time.monotonic()
    a=design(d,t,p)
    k,q=cached(d,t,p)[c['view']]
    y,center,scale,weights=target(d['y'][a['fit']],c)
    model=SVR(kernel='precomputed',C=c['C'],epsilon=0.,tol=SETTINGS['tolerance'],
              max_iter=SETTINGS['maximum_iterations'],cache_size=1024)
    model.fit(k,y,sample_weight=weights)
    assert model.fit_status_==0,'Optimizer did not converge; do not claim a completed comparison.'
    state=dict(signature=signature(),config=c,fit=a['fit'],query=a['query'],center=center,scale=scale,
               weights=weights,model=model,iterations=int(model.n_iter_))
    prediction=inverse(model.predict(q),state,t)
    assert np.isfinite(prediction).all()
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(state,folder/'state.joblib',compress=3)
    np.savez_compressed(file,signature=signature(),fit=a['fit'],query=a['query'],p=prediction)
    save(folder/'metadata.json',dict(signature=signature(),model_sha256=sha(folder/'state.joblib'),
         prediction_sha256=sha(file),iterations=int(model.n_iter_),converged=True,seconds=time.monotonic()-start))
    emit('expert_done',task=t,part=p,config=c['name'],iterations=int(model.n_iter_),seconds=time.monotonic()-start)
    return prediction


def replay(d,t,p,c):
    a=design(d,t,p)
    state=joblib.load(OUT/'tasks'/t/p/c['name']/'state.joblib')
    assert state['signature']==signature()
    for key in ('fit','query'):np.testing.assert_array_equal(state[key],a[key])
    path=OUT/'kernels'/t/(p+'.npz')
    z=np.load(path)
    assert str(z['signature'])==signature()
    return inverse(state['model'].predict(z[c['view']+'_query']),state,t)


def freeze():
    save(OUT/'protocol.json',dict(signature=signature(),tasks=TASKS,configs=CONFIGS,
         weights=WEIGHTS,settings=SETTINGS,all_fitting_remote=True,external_data=False,external_weights=False,
         test_labels_used=False,
         geometry='Six restored count histograms, L1-normalized then square rooted. 80percent mean polynomial Bhattacharyya or Gaussian Hellinger-map kernel; 15percent augmented RBF+5percent physical RBF.',
         loss='epsilon0 absolute residual SVR, exact convex objective to specified optimizer tolerance; no approximate early iterate used.',
         modes='Native labels for all9; weighted log1p alternative only on four nonnegative ranking tasks. Derivative compensation approximates native risk locally, not an exact equivalence.',
         development_fits=104,random_seeds='None: deterministic convex solver; no fake duplicate seed models.',
         selection='Complete all104 development fits then unique maxmin fold0/1 choice and weight. Freeze before fold2/valid, no fallback.',
         deployment='Same native-unit linear blend for MAE, Spearman, and CSV. Four MAEs positive plus four rank gains positive for sorting tasks.',
         limitations='Historical control partly proxy; four rank MAE references calibrated; public labels reused; no official score guarantee.'))
