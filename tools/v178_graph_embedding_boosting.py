"""Endpoint-specific boosting on frozen, globally query-excluded graph encoders."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v175_shared_graph_hybrid as graph
import v148_experiment_runner as runner
v, b, r83 = graph.v, graph.b, graph.r83
OUT = ROOT / 'admet_model_outputs_v178'
SEEDS = (1784026, 1784027)
CONFIGS = (dict(depth=3, trees=800), dict(depth=6, trees=1000))
METHOD = ('Frozen v175 shared graph/fingerprint representations from two independently seeded encoders, '
          'followed by task-specific XGBoost. Encoders exclude all query identities/scaffolds across '
          'every endpoint; booster fits only its target training rows. No measured-assay features, '
          'historical prediction features, new external data, or relaxed acceptance rules.')


def sig():
    return hashlib.sha256((b.sha(__file__) + graph.sig() + b.sha(runner.__file__)).encode()).hexdigest()


def key(c):
    return 'd' + str(c['depth']) + '_t' + str(c['trees'])


def initialize():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)


def embedding_design(d, part):
    a = graph.design(d, part)
    # Descriptor scales use only the encoder training molecules.
    queries = np.unique(np.concatenate(list(a['queries'].values())))
    requested = []
    for k in range(len(d['tasks'])):
        fit, q = v.split(d, k, part)
        assert not set(fit) & set(queries)
        requested.extend((fit, q))
    ids = np.unique(d['mi'][np.concatenate(requested)])
    return a, ids


def encode(net, d, a, ids):
    graphs = graph.graph.build_graphs(d)
    fp = torch.as_tensor(a['fp'], device='cuda')
    desc = torch.as_tensor(a['desc'], device='cuda')
    output = []
    net.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), 96):
            ix = ids[start:start + 96]
            output.append(net(*graph.graph.batch(graphs, ix, 'cuda'), fp[ix], desc[ix], 0).cpu().numpy())
    del fp, desc
    return np.concatenate(output).astype(np.float32)


def load_encoder(path, config):
    state = torch.load(path, weights_only=True)
    assert state['signature'] == graph.sig() and state['config'] == config
    net = graph.Net(state['ad'], state['ed'], state['dd'], state['tasks'], config).cuda()
    net.load_state_dict(state['state'])
    # Extract the existing shared layer without editing the trained source/model.
    net.heads[0] = torch.nn.Identity()
    return net


def embeddings(d, part, force=False):
    if not force and ('_v178_' + part) in d:
        return d['_v178_' + part]
    folder = OUT / 'features' / part
    path = folder / 'embeddings.npz'
    config = graph.CONFIGS[0]
    source = graph.OUT / 'models' / part / graph.key(config)
    if part == 'final' and not (source / 'predictions.npz').exists():
        confirmation = json.loads((graph.OUT / 'confirmation.json').read_text())
        assert confirmation['status'] == 'COMPLETE'
        graph.predict(d, 0, part, config)
    for _ in range(720):
        if (source / 'predictions.npz').exists():
            break
        time.sleep(15)
    else:
        raise TimeoutError('v175 encoder incomplete; preserve prior evidence')
    a, ids = embedding_design(d, part)
    models = {f'model_{seed}.pt': b.sha(source / f'model_{seed}.pt') for seed in graph.SEEDS}
    if path.exists() and not force:
        z = np.load(path)
        assert str(z['signature']) == sig()
        np.testing.assert_array_equal(ids, z['ids'])
        np.testing.assert_array_equal(a['source'], z['source'])
        assert json.loads(str(z['models'])) == models
        x = z['x'].copy()
    else:
        started = time.monotonic()
        parts = []
        for seed in graph.SEEDS:
            net = load_encoder(source / f'model_{seed}.pt', config)
            value = encode(net, d, a, ids)
            replay = load_encoder(source / f'model_{seed}.pt', config)
            # Full replay of representations, not just one diagnostic row.
            np.testing.assert_array_equal(value, encode(replay, d, a, ids))
            parts.append(value)
            del net, replay
            torch.cuda.empty_cache()
        x = np.concatenate(parts + [a['desc'][ids]], axis=1).astype(np.float32)
        assert np.isfinite(x).all()
        if not force:
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, signature=sig(), ids=ids, source=a['source'],
                                models=json.dumps(models), x=x)
        print(json.dumps(dict(stage='encode', part=part, rows=len(ids),
                              dimensions=x.shape[1], seconds=round(time.monotonic() - started))), flush=True)
    result = dict(ids=ids, source=a['source'], x=x, model_hashes=models)
    if not force:
        d['_v178_' + part] = result
    return result


def design(d, k, part):
    fit, q = v.split(d, k, part)
    e = embeddings(d, part)
    fi, qi = np.searchsorted(e['ids'], d['mi'][fit]), np.searchsorted(e['ids'], d['mi'][q])
    np.testing.assert_array_equal(e['ids'][fi], d['mi'][fit])
    np.testing.assert_array_equal(e['ids'][qi], d['mi'][q])
    return dict(fit=fit, query=q, x=e['x'][fi], qx=e['x'][qi], encoder_source=e['source'])


def predict(d, k, part, c):
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    path = folder / 'predictions.npz'
    if path.exists():
        z = np.load(path)
        assert str(z['signature']) == sig()
        np.testing.assert_array_equal(z['query'], v.split(d, k, part)[1])
        return z['p'].copy()
    a = design(d, k, part)
    y, metric = d['y'][a['fit']], d['metrics'][k]
    center = float(np.median(y)) if metric == 'MAE' else 0.
    scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if metric == 'MAE' else 1.
    target = (y - center) / scale if metric == 'MAE' else r83.ranknorm(y) if metric == 'Spearman' else y
    dm = xgb.DMatrix(a['x'], label=target, nthread=2)
    qm = xgb.DMatrix(a['qx'], nthread=2)
    folder.mkdir(parents=True, exist_ok=True)
    members = []
    started = time.monotonic()
    for seed in SEEDS:
        model = xgb.train(dict(objective='binary:logistic' if metric in ('AUROC', 'AUPRC')
                               else 'reg:absoluteerror' if metric == 'MAE' else 'reg:squarederror',
                               max_depth=c['depth'], eta=.025, reg_lambda=30, reg_alpha=.1,
                               min_child_weight=4, subsample=.85, colsample_bytree=.8,
                               max_bin=64, nthread=2, seed=seed, device='cuda', tree_method='hist',
                               verbosity=0), dm, num_boost_round=c['trees'])
        model.set_param({'device': 'cpu'})
        raw = model.predict(qm)
        mp = folder / f'model_{seed}.ubj'
        model.save_model(mp)
        replay = xgb.Booster()
        replay.load_model(mp)
        np.testing.assert_array_equal(raw, replay.predict(qm))
        members.append(raw)
    p = np.mean(members, axis=0).astype(float) * scale + center
    if d['tasks'][k] == 'ppbr_az':
        p = p.clip(0, 100)
    np.savez_compressed(path, signature=sig(), fit=a['fit'], query=a['query'],
                        encoder_source=a['encoder_source'], p=p, center=center, scale=scale)
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part, config=c,
                          seconds=round(time.monotonic() - started))), flush=True)
    return p


def smoke():
    initialize()
    d = v.load()
    a, ids = embedding_design(d, 'fold0')
    allq = np.unique(np.concatenate(list(a['queries'].values())))
    original = d['y'][allq].copy()
    try:
        d['y'][allq] = 777777.
        z, zi = embedding_design(d, 'fold0')
        np.testing.assert_array_equal(ids, zi)
        for key in ('source', 'target', 'median', 'center', 'scale', 'desc', 'fp'):
            np.testing.assert_array_equal(a[key], z[key])
    finally:
        d['y'][allq] = original
    source = graph.OUT / 'models/fold0' / graph.key(graph.CONFIGS[0])
    net = load_encoder(source / f'model_{graph.SEEDS[0]}.pt', graph.CONFIGS[0])
    p = encode(net, d, a, ids[:12])
    assert p.shape == (12, 192) and np.isfinite(p).all()
    OUT.mkdir(exist_ok=True)
    b.save(OUT / 'smoke.json', dict(status='PASS', source_sha256=b.sha(__file__),
                                    representation_dimensions=192, query_label_perturbation_invariant=True,
                                    global_query_scaffold_exclusion=True))
    print('SMOKE_PASS', flush=True)


def main():
    initialize()
    runner.run(sys.modules[__name__], v.load())


if __name__ == '__main__':
    smoke() if '--smoke' in sys.argv else main()
