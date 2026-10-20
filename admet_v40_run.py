"""Remote neural chemical-view learning with optional experimental FiLM conditioning."""
import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import admet_v24_models as b
import admet_v33_run as n
import admet_v37_run as chemistry_source
import admet_v39_run as previous

ROOT = n.ROOT
OUT = ROOT / 'admet_model_outputs_v40'
P = json.loads((ROOT / 'admet_v40_protocol.json').read_text(encoding='utf-8'))


def signature():
    names = ['admet_v40_run.py', 'admet_v40_protocol.json', 'admet_model_outputs_v39/analysis.json']
    return hashlib.sha256(json.dumps(dict(previous=previous.signature(),
        files={name: b.sha(ROOT / name) for name in names}), sort_keys=True).encode()).hexdigest()


def baseline(d):
    valid = np.load(chemistry_source.OUT / 'baseline_valid.npy').copy()
    cv = np.load(chemistry_source.OUT / 'baseline_cv_proxy.npy').copy()
    folds = np.load(b.OUT / 'folds.npy')
    report = json.loads((previous.OUT / 'analysis.json').read_text())
    assert report['signature'] == previous.signature()
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        k = d['tasks'].index(r['task'])
        m = r['metric']
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        vp, cp = np.zeros(len(vi)), np.zeros(len(tr))
        for source in s['sources']:
            v = previous.source_predictions(r['task'], source['version'], source['variant'], m, 'valid')
            c = np.load(previous.SOURCES[source['version']].OUT / r['task'] / 'cv_predictions.npy')[source['variant'], tr].copy()
            if s['name'] == 'complement_equal' and m != 'MAE':
                v = b.rank(v)
                for fold in range(3):
                    active = folds[tr] == fold
                    c[active] = b.rank(c[active])
            vp += source['weight'] * v
            cp += source['weight'] * c
        pred = b.blend(m, valid[vi], vp, s['weight'])
        np.testing.assert_allclose(b.delta(m, d['y'][vi], valid[vi], pred), s['valid_gain'], rtol=1e-8, atol=1e-10)
        valid[vi] = pred
        for fold in range(3):
            active = folds[tr] == fold
            ix = tr[active]
            cv[ix] = b.blend(m, cv[ix], cp[active], s['weight'])
    np.save(OUT / 'baseline_valid.npy', valid)
    np.save(OUT / 'baseline_cv_proxy.npy', cv)
    return valid, cv


def features(d, k, rows, query, splits, other, chemical):
    x, pre, _, target, audit = n.features(d, k, rows, query, splits, other)
    aux_pre = b.Preprocessor().fit(x['fit']['aux'])
    parts = {}
    for name, rr in [('fit', rows), ('query', query)]:
        mol = d['mi'][rr]
        parts[name] = [x[name]['fp'], chemical[mol, :4096], chemical[mol, 4096:],
                       x[name]['dense'], aux_pre.transform(x[name]['aux'])]
    audit.update(full_extended_fingerprints=True, extended_selection_label_free=True,
                 assay_preprocessor_fit_only=True, dense_preprocessor_fit_only=True)
    return parts, pre, aux_pre, target, audit


class ChemicalNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        h = P['hidden']
        drop = P['dropout']
        def layer(a, z):
            return nn.Sequential(nn.Linear(a, z), nn.LayerNorm(z), nn.SiLU(), nn.Dropout(drop))
        self.encoders = nn.ModuleList([layer(dim, h) for dim in dims[:4]])
        self.gate = nn.Sequential(nn.Linear(4*h, h), nn.SiLU(), nn.Linear(h, 4))
        self.residuals = nn.ModuleList([layer(h, h) for _ in range(2)])
        self.structure_head = nn.Sequential(layer(h, 64), nn.Linear(64, 1))
        self.assay = layer(dims[4], h)
        self.film = nn.Linear(h, 2*h)
        self.conditioned_head = nn.Sequential(layer(2*h, h), nn.Linear(h, 1))

    def forward(self, views):
        encoded = torch.stack([enc(x) for enc, x in zip(self.encoders, views[:4])], dim=1)
        weights = self.gate(encoded.flatten(1)).softmax(1)
        h = (encoded * weights.unsqueeze(-1)).sum(1)
        for layer in self.residuals:
            h = h + layer(h)
        structural = self.structure_head(h).squeeze(-1)
        evidence = views[4]
        if self.training:
            evidence = evidence * (torch.rand((len(evidence), 1), device=evidence.device) > .20)
        aux = self.assay(evidence)
        scale, offset = self.film(aux).chunk(2, dim=1)
        conditioned = h * (1 + .25 * scale.tanh()) + .25 * offset
        assisted = self.conditioned_head(torch.cat([conditioned, aux], dim=1)).squeeze(-1)
        return torch.stack([structural, assisted])


def objective(pred, y, weights, metric):
    total = pred.new_zeros(())
    for p in pred:
        if metric in ['AUROC', 'AUPRC']:
            losses = nn.functional.binary_cross_entropy_with_logits(p, y, reduction='none')
            positives, negatives = p[y > .5], p[y < .5]
            pair = .05 * nn.functional.softplus(negatives[None] - positives[:, None]).mean() if len(positives) and len(negatives) else 0.
        elif metric == 'MAE':
            losses = (p-y).abs()
            pair = 0.
        else:
            losses = nn.functional.smooth_l1_loss(p, y, beta=.3, reduction='none')
            diff = y[:, None] - y[None]
            active = diff.abs() > .15
            pair = .10 * nn.functional.softplus(-diff[active].sign() * (p[:, None]-p[None])[active]).mean() if active.any() else 0.
        total += (losses*weights).sum()/weights.sum() + pair
    return total*.5


def infer(net, views):
    net.eval()
    with torch.inference_mode():
        return np.concatenate([net([x[start:start+512] for x in views]).cpu().numpy()
            for start in range(0, len(views[0]), 512)], axis=1)


def decode(p, target, metric):
    if metric == 'Spearman':
        return p
    return target.decode(p)


def fit(d, k, rows, query, splits, other, chemical, dest):
    dest.mkdir(parents=True, exist_ok=True)
    sig = signature()
    cache = dest / 'predictions.npz'
    if cache.exists():
        z = np.load(cache)
        assert str(z['signature']) == sig
        np.testing.assert_array_equal(z['fit'], rows)
        np.testing.assert_array_equal(z['query'], query)
        return z['p'].copy()
    t0 = time.time()
    x, pre, aux_pre, target, audit = features(d, k, rows, query, splits, other, chemical)
    xx = {name: [torch.tensor(a, device='cuda') for a in value] for name, value in x.items()}
    y = torch.tensor(target.encode(d['y'][rows]), device='cuda')
    _, inv, counts = np.unique(d['canonical'][d['mi'][rows]], return_inverse=True, return_counts=True)
    weights = torch.tensor(1/np.sqrt(counts[inv]), dtype=torch.float32, device='cuda')
    dims = [a.shape[1] for a in xx['fit']]
    metric = d['metrics'][k]
    seeds = []
    for seed in P['seeds']:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = ChemicalNet(dims).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=P['learning_rate'], weight_decay=P['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, P['epochs'], eta_min=2e-5)
        snaps = []
        for epoch in range(1, P['epochs']+1):
            net.train()
            order = rng.permutation(len(rows))
            for start in range(0, len(order), P['batch_size']):
                ix = order[start:start+P['batch_size']]
                loss = objective(net([a[ix] for a in xx['fit']]), y[ix], weights[ix], metric)
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 3.)
                opt.step()
            scheduler.step()
            if epoch in P['snapshots']:
                pred = infer(net, xx['query'])
                path = dest / f'net_{seed}_{epoch}.pt'
                torch.save(dict(state={name: p.detach().cpu().clone() for name, p in net.state_dict().items()},
                    dims=dims, target=target, preprocessor=pre, assay_preprocessor=aux_pre, signature=sig), path)
                loaded = torch.load(path, map_location='cpu', weights_only=False)
                net.load_state_dict(loaded['state'])
                np.testing.assert_allclose(pred, infer(net, xx['query']), rtol=1e-5, atol=1e-5)
                snaps.append(decode(pred, target, metric))
        seeds.append(np.mean(snaps, axis=0))
        del net
        b.emit('v40_seed', task=d['tasks'][k], partition=str(dest.relative_to(OUT)), seed=seed, seconds=round(time.time()-t0))
    p = np.mean(seeds, axis=0)
    assert np.isfinite(p).all()
    np.savez_compressed(cache, p=p, fit=rows, query=query, signature=sig)
    audit.update(signature=sig, task=d['tasks'][k], fit_rows=len(rows), query_rows=len(query),
        seconds=time.time()-t0, model_reload_passed=True, test_labels_used=False, local_gpu_training=False)
    b.save(dest / 'metadata.json', audit)
    del xx, y, weights
    torch.cuda.empty_cache()
    return p


def variants(p, metric):
    values = p.copy() if metric == 'MAE' else np.stack([b.rank(a) for a in p])
    return np.concatenate([p, np.mean(values, axis=0)[None]], axis=0)


def screen(d, chemical, other):
    base, _ = baseline(d)
    records = []
    for k, task in enumerate(d['tasks']):
        tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        metric = d['metrics'][k]
        vp = variants(fit(d, k, tr, vi, [0], other, chemical, OUT / task / 'valid'), metric)
        options = [dict(variant=v, weight=w, valid_gain=b.delta(metric, d['y'][vi], base[vi], b.blend(metric, base[vi], p, w)))
            for v, p in enumerate(vp) for w in P['weights']]
        shortlisted = sorted({x['variant'] for x in options if x['valid_gain'] > .0002})
        records.append(dict(task=task, metric=metric, valid_rows=len(vi), valid_baseline=b.metric(metric, d['y'][vi], base[vi]),
            candidate_metrics=[b.metric(metric, d['y'][vi], p) for p in vp], screen_gain=max(x['valid_gain'] for x in options),
            shortlisted_variants=shortlisted, screen_grid=options))
        b.save(OUT / 'screen.json', dict(signature=signature(), status='IN_PROGRESS', tasks=records))
        b.emit('v40_screen', task=task, gain=records[-1]['screen_gain'], variants=shortlisted)
    b.save(OUT / 'screen.json', dict(signature=signature(), status='FROZEN', tasks=records,
        validation_reused=True, official_score_prediction=None))


def evaluate(d, chemical, other):
    prior = json.loads((OUT / 'screen.json').read_text())
    assert prior['status'] == 'FROZEN' and prior['signature'] == signature()
    base = np.load(OUT / 'baseline_valid.npy')
    old = np.load(OUT / 'baseline_cv_proxy.npy')
    folds = np.load(b.OUT / 'folds.npy')
    records = []
    for r in prior['tasks']:
        k = d['tasks'].index(r['task'])
        metric = r['metric']
        tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        record = dict(**r, selection=None, grid=[])
        if r['shortlisted_variants']:
            cp = np.full((3, len(d['y'])), np.nan)
            for fold in range(3):
                q = tr[folds[tr] == fold]
                rows = tr[folds[tr] != fold]
                cp[:, q] = variants(fit(d, k, rows, q, [0], other, chemical, OUT / r['task'] / f'fold{fold}'), metric)
            np.save(OUT / r['task'] / 'cv_predictions.npy', cp)
            vp = variants(np.load(OUT / r['task'] / 'valid/predictions.npz')['p'], metric)
            grid = []
            for v in r['shortlisted_variants']:
                for w in P['weights']:
                    gains = []
                    for fold in range(3):
                        ix = tr[folds[tr] == fold]
                        gains.append(b.delta(metric, d['y'][ix], old[ix], b.blend(metric, old[ix], cp[v, ix], w)))
                    vg = b.delta(metric, d['y'][vi], base[vi], b.blend(metric, base[vi], vp[v], w))
                    obj = float(.45*vg + .55*np.mean(gains) - .04*np.std(gains))
                    grid.append(dict(variant=v, weight=w, valid_gain=vg, cv_gains=gains, cv_mean=float(np.mean(gains)),
                        objective=obj, eligible=bool(vg > .0002 and np.mean(gains) > 0 and obj > .0005)))
            choices = [x for x in grid if x['eligible']]
            record.update(grid=grid, selection=max(choices, key=lambda x: x['objective']) if choices else None)
        records.append(record)
        b.save(OUT / 'analysis.json', dict(signature=signature(), status='IN_PROGRESS', tasks=records))
        b.emit('v40_selection', task=r['task'], selection=record['selection'])
    b.save(OUT / 'analysis.json', dict(signature=signature(), status='EVALUATED', baseline='v39', baseline_score=P['baseline_score'],
        tasks=records, selected_tasks=[r['task'] for r in records if r['selection']], official_score_prediction=None, limitations=P['limitations']))


def final(d, chemical):
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
        compact[r['task']] = fit(d, k, rows, q, [0, 1], other, chemical, OUT / r['task'] / 'final').tolist()
    b.save(OUT / 'test_predictions_compact.json', compact)


def export(d):
    report = json.loads((OUT / 'analysis.json').read_text())
    assert report['signature'] == signature()
    if not report['selected_tasks']:
        b.emit('v40_no_candidate')
        return
    source = next((ROOT / 'admet_submissions/v39').glob('*.csv'))
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
    dest = ROOT / 'admet_submissions/v40'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / '我从山北来带着兰花草-小分子药物预测技术创新赛-20261008.csv'
    buf = io.StringIO(newline='')
    csv.writer(buf, lineterminator='\r\n').writerows(rows)
    path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'))
    assert read(path) == rows
    b.save(OUT / 'source_manifest.json', {name: b.sha(ROOT / name) for name in
        ['admet_v40_run.py', 'admet_v40_protocol.json', 'admet_v40_remote_verify.py']})
    b.save(dest / 'verification.json', dict(status='EXPERIMENTAL_AWAITING_PLATFORM', signature=signature(), baseline='v39',
        baseline_score=P['baseline_score'], baseline_sha256=P['baseline_sha256'], sha256=b.sha(path), rows=16379, tasks=22,
        columns=6, changed_tasks=changed, bytes=path.stat().st_size, encoding='UTF-8 BOM', line_endings='CRLF',
        template_identity_exact=True, training_host=P['training_host'], local_gpu_training=False,
        automatically_uploaded=False, platform_score=None, limitations=P['limitations']))
    b.emit('v40_export', sha256=b.sha(path), changed_tasks=changed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'evaluate', 'final', 'export', 'all'])
    args = parser.parse_args()
    b.remote_only()
    assert ROOT == Path(P['remote_root'])
    torch.backends.cuda.matmul.allow_tf32 = True
    OUT.mkdir(parents=True, exist_ok=True)
    d = b.build_data()
    chemical = chemistry_source.chemistry(d)
    if args.stage in ['screen', 'all']:
        screen(d, chemical, n.other_cache(d, [0]))
    if args.stage in ['evaluate', 'all']:
        evaluate(d, chemical, n.other_cache(d, [0]))
    if args.stage in ['final', 'all']:
        final(d, chemical)
    if args.stage in ['export', 'all']:
        export(d)
    b.save(OUT / f'{args.stage}_completed.json', dict(signature=signature(), completed=True))


if __name__ == '__main__':
    main()
