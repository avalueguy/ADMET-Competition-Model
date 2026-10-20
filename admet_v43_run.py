"""Partition-isolated learned graph representations and task-specific boosting."""
import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import torch
from xgboost import XGBClassifier, XGBRegressor
import admet_v24_models as b
import admet_v27_models as graph
import admet_v33_run as n
import admet_v41_run as parent
import admet_v43_audit as audit_source

ROOT = n.ROOT
OUT = ROOT / 'admet_model_outputs_v43'
P = json.loads((ROOT / 'admet_v43_protocol.json').read_text())


def signature():
    names = ['admet_v43_run.py', 'admet_v43_protocol.json', 'admet_v43_audit.py', 'admet_model_outputs_v42/analysis.json']
    return hashlib.sha256(json.dumps(dict(graph=graph.signature(), parent=parent.signature(),
        files={name: b.sha(ROOT / name) for name in names}), sort_keys=True).encode()).hexdigest()


def source_partition(dest):
    name = dest.name
    if name == 'valid':
        return 'valid'
    if name == 'final':
        return 'final'
    assert name in ['fold0', 'fold1', 'fold2']
    return 'cv/' + name


def latent_forward(net, packed):
    atom, bond, src, dst, rev, mol, natoms = packed
    initial = torch.nn.functional.silu(net.edge(torch.cat([atom[src], bond], 1)))
    h = initial
    for _ in range(graph.P['depth'] - 1):
        incoming = h.new_zeros((len(atom), graph.P['hidden']))
        incoming.index_add_(0, dst, h)
        h = net.drop(torch.nn.functional.silu(initial + net.message(incoming[src] - h[rev])))
    incoming = h.new_zeros((len(atom), graph.P['hidden']))
    incoming.index_add_(0, dst, h)
    atoms = torch.nn.functional.silu(net.atom(torch.cat([atom, incoming], 1)))
    summed = atoms.new_zeros((len(natoms), graph.P['hidden']))
    summed.index_add_(0, mol, atoms)
    return net.pool(torch.cat([summed / natoms[:, None], summed / natoms[:, None].sqrt(), natoms.log1p()[:, None]], 1))


_LATENT = {}
def representations(d, g, partition):
    if partition in _LATENT:
        return _LATENT[partition]
    dest = OUT / 'features' / partition.replace('/', '_')
    dest.mkdir(parents=True, exist_ok=True)
    sources = [graph.OUT / partition / str(seed) / 'checkpoint_5500.pt' for seed in P['graph_seeds']]
    meta = dict(signature=signature(), sources={str(p.relative_to(ROOT)): b.sha(p) for p in sources},
        source_graph_signature=graph.signature(), labels_at_feature_extraction=False)
    path = dest / 'latent.npz'
    if path.exists():
        assert json.loads((dest / 'latent.json').read_text()) == meta
        _LATENT[partition] = np.load(path)['x'].copy()
        return _LATENT[partition]
    parts = []
    for source in sources:
        ck = torch.load(source, map_location='cpu', weights_only=False)
        assert ck['signature'] == graph.signature()
        net = graph.BondNet(*ck['dimensions']).cuda().eval()
        net.load_state_dict(ck['state'])
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(d['smiles']), 128):
                mm = np.arange(start, min(start + 128, len(d['smiles'])))
                embeddings.append(latent_forward(net, graph.pack(g, mm)).cpu().numpy())
        parts.append(np.concatenate(embeddings))
        del net
        b.emit('v43_latent', partition=partition, source=str(source.relative_to(ROOT)))
    x = np.concatenate(parts, axis=1)
    assert x.shape == (len(d['smiles']), 256) and np.isfinite(x).all()
    np.savez_compressed(path, x=x)
    b.save(dest / 'latent.json', meta)
    _LATENT[partition] = x
    torch.cuda.empty_cache()
    return x


def representation_audit(d, rows, query, partition):
    source = graph.OUT / partition / str(P['graph_seeds'][0]) / 'predictions.npz'
    z = np.load(source)
    assert str(z['signature']) == graph.signature()
    source_rows = z['fit']
    assert np.all(d['split'][source_rows] < 2)
    overlap = len(set(d['rg'][source_rows]) & set(d['rg'][query]))
    identity_overlap = len(set(d['canonical'][d['mi'][source_rows]]) & set(d['canonical'][d['mi'][query]]))
    if partition != 'final':
        assert overlap == identity_overlap == 0
        # A representation may be trained on a subset of downstream fitting rows,
        # but never on a queried target task label.
        assert not set(source_rows) & set(query)
    return dict(graph_partition=partition, source_fit_rows=len(source_rows),
        queried_target_labels_seen_by_graph=0, graph_source_query_group_overlap=overlap,
        graph_source_query_identity_overlap=identity_overlap, graph_source_signature=graph.signature())


def features(d, k, rows, query, splits, other, chem, latent):
    parts, target, meta = parent.features(d, k, rows, query, splits, other, chem)
    raw = np.concatenate([d['desc'], d['emb']], axis=1)
    unique = np.unique(d['mi'][rows])
    dense_pre = b.Preprocessor().fit(raw[unique])
    graph_pre = b.Preprocessor().fit(latent[unique])
    dense = dense_pre.transform(raw)
    learned = graph_pre.transform(latent)
    x = {}
    for name, rr in [('fit', rows), ('query', query)]:
        mol = d['mi'][rr]
        control = parts[name][1]
        x[name] = [control, np.concatenate([dense[mol], learned[mol]], 1),
                   np.concatenate([control, learned[mol]], 1)]
    meta.update(graph_preprocessor_fit_only=True, latent_dim=latent.shape[1],
        latent_seed_coordinates_concatenated=True, graph_extraction_labels_used=False)
    return x, target, meta, (dense_pre, graph_pre)


def make_model(metric, seed, depth, trees):
    cls = metric in ['AUROC', 'AUPRC']
    return (XGBClassifier if cls else XGBRegressor)(n_estimators=trees, max_depth=depth,
        learning_rate=.035, min_child_weight=4, subsample=.85, colsample_bytree=.8,
        reg_lambda=18., reg_alpha=.10, tree_method='hist', device='cuda', max_bin=128,
        n_jobs=3, random_state=seed, objective='binary:logistic' if cls else (
            'reg:absoluteerror' if metric == 'MAE' else 'reg:squarederror'))


def predict_model(model, x, metric, target):
    p = model.predict_proba(x)[:, 1] if metric in ['AUROC', 'AUPRC'] else model.predict(x)
    return target.decode(p) if metric == 'MAE' else p


def fit(d, g, k, rows, query, splits, other, chem, dest):
    dest.mkdir(parents=True, exist_ok=True)
    sig = signature()
    path = dest / 'predictions.npz'
    if path.exists():
        z = np.load(path)
        assert str(z['signature']) == sig
        np.testing.assert_array_equal(z['fit'], rows)
        np.testing.assert_array_equal(z['query'], query)
        return z['p'].copy()
    partition = source_partition(dest)
    t0 = time.time()
    isolation = representation_audit(d, rows, query, partition)
    latent = representations(d, g, partition)
    x, target, meta, pre = features(d, k, rows, query, splits, other, chem, latent)
    metric = d['metrics'][k]
    y = target.encode(d['y'][rows])
    _, inv, count = np.unique(d['canonical'][d['mi'][rows]], return_inverse=True, return_counts=True)
    weights = 1 / np.sqrt(count[inv])
    pred = []
    for family in range(3):
        parts = []
        for seed, depth, trees in zip(P['seeds'], P['depths'], P['estimators']):
            model = make_model(metric, seed, depth, trees)
            model.fit(x['fit'][family], y, sample_weight=weights)
            pathm = dest / f'family{family}_{seed}.ubj'
            model.save_model(pathm)
            p = predict_model(model, x['query'][family], metric, target)
            restored = make_model(metric, seed, depth, trees)
            restored.load_model(pathm)
            restored.set_params(device='cuda', n_jobs=3)
            np.testing.assert_allclose(p, predict_model(restored, x['query'][family], metric, target), rtol=2e-6, atol=2e-6)
            parts.append(p)
        pred.append(np.mean(parts, 0))
    p = np.stack(pred)
    assert np.isfinite(p).all()
    np.savez_compressed(path, p=p, fit=rows, query=query, signature=sig)
    # Persist only preprocessing, labels are neither query inputs nor artifacts here.
    torch.save(dict(target=target, dense_pre=pre[0], graph_pre=pre[1], signature=sig), dest / 'preprocessing.pt')
    meta.update(isolation, signature=sig, task=d['tasks'][k], fit_rows=len(rows), query_rows=len(query),
        seconds=time.time()-t0, model_reload_passed=True, test_labels_used=False, local_gpu_training=False)
    b.save(dest / 'metadata.json', meta)
    b.emit('v43_fit', task=d['tasks'][k], partition=partition, seconds=round(time.time()-t0))
    return p


def variants(p, metric):
    values = p.copy() if metric == 'MAE' else np.stack([b.rank(a) for a in p])
    return np.concatenate([p, (.5*values[1]+.5*values[2])[None]], 0)


def screen(d, g, chem, other):
    base, _ = audit_source.baseline(d)
    records = []
    for k, task in enumerate(d['tasks']):
        rows = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        query = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        metric = d['metrics'][k]
        vp = variants(fit(d, g, k, rows, query, [0], other, chem, OUT / task / 'valid'), metric)
        options = [dict(variant=v, weight=w, valid_gain=b.delta(metric, d['y'][query], base[query], b.blend(metric, base[query], p, w)))
                   for v, p in enumerate(vp) for w in P['weights']]
        # Low-dimensional graph representations may offer scaffold gains even when
        # the idiosyncratic official valid is slightly worse. Freeze once here.
        shortlisted = sorted({x['variant'] for x in options if x['valid_gain'] > .0002})
        records.append(dict(task=task, metric=metric, valid_rows=len(query),
            valid_baseline=b.metric(metric, d['y'][query], base[query]),
            candidate_metrics=[b.metric(metric, d['y'][query], p) for p in vp],
            screen_gain=max(x['valid_gain'] for x in options), shortlisted_variants=shortlisted, screen_grid=options))
        b.save(OUT / 'screen.json', dict(signature=signature(), status='IN_PROGRESS', tasks=records))
        b.emit('v43_screen', task=task, gain=records[-1]['screen_gain'], variants=shortlisted)
    b.save(OUT / 'screen.json', dict(signature=signature(), status='FROZEN', tasks=records,
        validation_reused=True, official_score_prediction=None))


def domain_cv_gain(d, metric, ix, old, candidate, cyclic, test_cyclic_fraction):
    full = b.delta(metric, d['y'][ix], old, candidate)
    strata = []
    for flag, weight in [(False, 1-test_cyclic_fraction), (True, test_cyclic_fraction)]:
        active = cyclic[d['mi'][ix]] == flag
        if np.sum(active) < 20 or (metric in ['AUROC', 'AUPRC'] and len(np.unique(d['y'][ix[active]])) < 2):
            continue
        value = b.delta(metric, d['y'][ix[active]], old[active], candidate[active])
        if np.isfinite(value) and weight > 0:
            strata.append((value, weight))
    domain = sum(g*w for g, w in strata)/sum(w for _, w in strata) if strata else full
    return float(.5*full+.5*domain), float(full), float(domain)


def evaluate(d, g, chem, other):
    prior = json.loads((OUT / 'screen.json').read_text())
    assert prior['status'] == 'FROZEN' and prior['signature'] == signature()
    base = np.load(OUT / 'baseline_valid.npy')
    old = np.load(OUT / 'baseline_cv_proxy.npy')
    folds = np.load(b.OUT / 'folds.npy')
    cyclic = np.array([not str(a).startswith('ACYCLIC:') for a in d['groups']])
    records = []
    for r in prior['tasks']:
        k = d['tasks'].index(r['task'])
        metric = r['metric']
        tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        te = np.flatnonzero((d['tid'] == k) & (d['split'] == 2))
        vf, tf = float(cyclic[d['mi'][vi]].mean()), float(cyclic[d['mi'][te]].mean())
        valid_weight = .15 if abs(vf-tf) > .5 or len(set(d['rg'][vi])) < 2 else .45
        record = dict(**r, selection=None, grid=[], valid_cyclic_fraction=vf, test_cyclic_fraction=tf,
            selection_valid_weight=valid_weight)
        if r['shortlisted_variants']:
            cp = np.full((4, len(d['y'])), np.nan)
            for fold in range(3):
                q = tr[folds[tr] == fold]
                rows = tr[folds[tr] != fold]
                cp[:, q] = variants(fit(d, g, k, rows, q, [0], other, chem, OUT / r['task'] / f'fold{fold}'), metric)
            np.save(OUT / r['task'] / 'cv_predictions.npy', cp)
            vp = variants(np.load(OUT / r['task'] / 'valid/predictions.npz')['p'], metric)
            grid = []
            for v in r['shortlisted_variants']:
                for w in P['weights']:
                    gains, fulls, domains = [], [], []
                    for fold in range(3):
                        ix = tr[folds[tr] == fold]
                        new = b.blend(metric, old[ix], cp[v, ix], w)
                        gain, full, domain = domain_cv_gain(d, metric, ix, old[ix], new, cyclic, tf)
                        gains.append(gain); fulls.append(full); domains.append(domain)
                    vg = b.delta(metric, d['y'][vi], base[vi], b.blend(metric, base[vi], vp[v], w))
                    obj = float(valid_weight*vg+(1-valid_weight)*np.mean(gains)-.04*np.std(gains))
                    grid.append(dict(variant=v, weight=w, valid_gain=vg, cv_gains=fulls, domain_cv_gains=domains,
                        adjusted_cv_gains=gains, cv_mean=float(np.mean(fulls)), adjusted_cv_mean=float(np.mean(gains)),
                        objective=obj, eligible=bool(vg > -.005 and np.mean(gains) > .0005 and obj > .0005)))
            choices = [x for x in grid if x['eligible']]
            record.update(grid=grid, selection=max(choices, key=lambda x: x['objective']) if choices else None)
        records.append(record)
        b.save(OUT / 'analysis.json', dict(signature=signature(), status='IN_PROGRESS', tasks=records))
        b.emit('v43_selection', task=r['task'], selection=record['selection'])
    b.save(OUT / 'analysis.json', dict(signature=signature(), status='EVALUATED', baseline='v42', baseline_score=P['baseline_score'],
        tasks=records, selected_tasks=[r['task'] for r in records if r['selection']],
        official_score_prediction=None, limitations=P['limitations']))


def final(d, g, chem):
    report = json.loads((OUT / 'analysis.json').read_text())
    assert report['status'] == 'EVALUATED' and report['signature'] == signature()
    other = n.other_cache(d, [0, 1])
    compact = {}
    for r in report['tasks']:
        if r['selection'] is None:
            continue
        k = d['tasks'].index(r['task'])
        rows = np.flatnonzero((d['tid'] == k) & (d['split'] < 2))
        q = np.flatnonzero((d['tid'] == k) & (d['split'] == 2))
        compact[r['task']] = fit(d, g, k, rows, q, [0, 1], other, chem, OUT / r['task'] / 'final').tolist()
    b.save(OUT / 'test_predictions_compact.json', compact)


def export(d):
    report = json.loads((OUT / 'analysis.json').read_text())
    assert report['signature'] == signature()
    if not report['selected_tasks']:
        b.emit('v43_no_candidate')
        return
    source = next((ROOT / 'admet_submissions/v42').glob('*.csv'))
    assert b.sha(source) == P['baseline_sha256']
    def read(path):
        return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'), newline='')))
    rows = read(source)
    template = read(ROOT / 'admet_official/submission_template.csv')
    compact = json.loads((OUT / 'test_predictions_compact.json').read_text())
    changed = {}
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        pos = [i for i, row in enumerate(rows) if i > 0 and row[0] == r['task']]
        old = np.array([float(rows[i][4]) for i in pos])
        p = variants(np.array(compact[r['task']]), r['metric'])[s['variant']]
        assert len(p) == len(old)
        merged = b.blend(r['metric'], old, p, s['weight'])
        for i, value in zip(pos, merged):
            rows[i][4] = repr(float(value))
        changed[r['task']] = int(np.sum(old != merged))
    assert changed and len(rows) == len(template) == 16380 and rows[0] == template[0]
    for row, target in zip(rows[1:], template[1:]):
        assert len(row) == 6 and row[:4]+row[5:] == target[:4]+target[5:] and np.isfinite(float(row[4]))
        if row[5] in ['AUROC', 'AUPRC']:
            assert 0 <= float(row[4]) <= 1
    dest = ROOT / 'admet_submissions/v43'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / '我从山北来带着兰花草-小分子药物预测技术创新赛-20261008.csv'
    buf = io.StringIO(newline='')
    csv.writer(buf, lineterminator='\r\n').writerows(rows)
    path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'))
    assert read(path) == rows
    b.save(OUT / 'source_manifest.json', {name: b.sha(ROOT / name) for name in
        ['admet_v43_run.py', 'admet_v43_protocol.json', 'admet_v43_audit.py', 'admet_v43_remote_verify.py']})
    b.save(dest / 'verification.json', dict(status='EXPERIMENTAL_AWAITING_PLATFORM', signature=signature(), baseline='v42',
        baseline_score=P['baseline_score'], baseline_sha256=P['baseline_sha256'], sha256=b.sha(path), rows=16379, tasks=22,
        columns=6, changed_tasks=changed, bytes=path.stat().st_size, encoding='UTF-8 BOM', line_endings='CRLF',
        template_identity_exact=True, training_host=P['training_host'], local_gpu_training=False,
        automatically_uploaded=False, platform_score=None, limitations=P['limitations']))
    b.emit('v43_export', sha256=b.sha(path), changed_tasks=changed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'evaluate', 'final', 'export', 'all'])
    args = parser.parse_args()
    b.remote_only()
    assert ROOT == Path(P['remote_root'])
    # Disable TF32 for deterministic graph feature extraction and subsequent replays.
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    d = b.build_data()
    g = graph.graphs(d)
    chem = parent.chemistry(d)
    if args.stage in ['screen', 'all']:
        screen(d, g, chem, n.other_cache(d, [0]))
    if args.stage in ['evaluate', 'all']:
        evaluate(d, g, chem, n.other_cache(d, [0]))
    if args.stage in ['final', 'all']:
        final(d, g, chem)
    if args.stage in ['export', 'all']:
        export(d)
    b.save(OUT / f'{args.stage}_completed.json', dict(signature=signature(), completed=True))


if __name__ == '__main__':
    main()
