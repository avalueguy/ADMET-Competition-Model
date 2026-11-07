"""Independently replay frozen graph representations and v178 tree ensembles."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v178_graph_embedding_boosting as model
import v148_experiment_runner as runner
from finalize_v124_local import rows as table
from repack_upload_template import field_spans
from verify_submission import verify
v, b, r83 = model.v, model.b, model.r83
OUT = ROOT / 'admet_model_outputs_v179'


def main():
    model.initialize()
    OUT.mkdir(exist_ok=True)
    cp = model.OUT / 'confirmation.json'
    for _ in range(720):
        if cp.exists() and json.loads(cp.read_text())['status'] == 'COMPLETE':
            break
        time.sleep(15)
    else:
        raise TimeoutError('v178 incomplete; preserve partial models')
    confirmation = json.loads(cp.read_text())
    frozen = json.loads((model.OUT / 'frozen_selection.json').read_text())
    assert confirmation['signature'] == frozen['signature'] == model.sig()
    selected = {r['task']: r for r in confirmation['records'] if r['accepted']}
    if not selected:
        b.save(OUT / 'no_candidate.json', dict(status='NO_CONFIRMED_NEW_MODEL',
               training_version='v178', baseline='v145', official_baseline_score=51.166,
               submission_created=False, confirmation_sha256=b.sha(cp)))
        print('NO_CONFIRMED_NEW_MODEL', flush=True)
        return
    d = v.load()
    checks = []
    replayed = {}
    for part in v.PARTS:
        a, ids = model.embedding_design(d, part)
        cached = model.embeddings(d, part)
        independent = model.embeddings(d, part, force=True)
        for key in ('ids', 'source', 'x'):
            np.testing.assert_array_equal(cached[key], independent[key])
        allq = np.unique(np.concatenate(list(a['queries'].values())))
        labels = d['y'][allq].copy()
        try:
            d['y'][allq] = 999999.
            z, zi = model.embedding_design(d, part)
            np.testing.assert_array_equal(ids, zi)
            for key in ('source', 'target', 'median', 'center', 'scale', 'desc', 'fp'):
                np.testing.assert_array_equal(a[key], z[key])
        finally:
            d['y'][allq] = labels
        for task, record in selected.items():
            k = d['tasks'].index(task)
            design = model.design(d, k, part)
            folder = model.OUT / 'models' / task / part / model.key(record['config'])
            saved = np.load(folder / 'predictions.npz')
            assert str(saved['signature']) == model.sig()
            for key in ('fit', 'query', 'encoder_source'):
                np.testing.assert_array_equal(design[key], saved[key])
            y = d['y'][design['fit']]
            center = float(np.median(y)) if record['metric'] == 'MAE' else 0.
            scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if record['metric'] == 'MAE' else 1.
            assert center == float(saved['center']) and scale == float(saved['scale'])
            members, hashes = [], {}
            for seed in model.SEEDS:
                path = folder / f'model_{seed}.ubj'
                booster = xgb.Booster()
                booster.load_model(path)
                # UBJ stores trees; generic training parameters are not restored.
                assert booster.num_boosted_rounds() == record['config']['trees']
                members.append(booster.predict(xgb.DMatrix(design['qx'], nthread=2)))
                hashes[path.name] = b.sha(path)
            p = np.mean(members, axis=0).astype(float) * scale + center
            if task == 'ppbr_az':
                p = p.clip(0, 100)
            np.testing.assert_array_equal(p, saved['p'])
            replayed[(task, part)] = p
            checks.append(dict(task=task, part=part, model_reload=True,
                               query_label_perturbation_invariant=True,
                               independent_graph_representations_replayed=True,
                               confirmation_scaffolds_excluded=part != 'final',
                               model_hashes=hashes, encoder_hashes=cached['model_hashes'],
                               embedding_sha256=b.sha(model.OUT / 'features' / part / 'embeddings.npz')))
        print(json.dumps(dict(stage='independent_replay', part=part, tasks=list(selected))), flush=True)
    original = table(v.ANCHOR)
    rows = [r.copy() for r in original]
    for task, record in selected.items():
        k = d['tasks'].index(task)
        choice = next(r for r in frozen['selected'] if r['task'] == task)
        assert runner.confirm(d, k, model.predict, choice) == record
        for part in ('fold0', 'fold1', 'fold2', 'valid'):
            q = v.split(d, k, part)[1]
            p = v.mix(record['metric'], v.baseline(d, k, part), replayed[(task, part)], record['weight'])
            expected = next(r for r in record['confirmation'] if r['part'] == part)
            np.testing.assert_allclose(r83.score(record['metric'], d['y'][q], p), expected['candidate'], atol=1e-12, rtol=0)
        p = v.mix(record['metric'], v.baseline(d, k, 'final'), replayed[(task, 'final')], record['weight'])
        positions = [i for i, r in enumerate(original) if i and r[0] == task]
        assert len(positions) == len(p)
        for i, value in zip(positions, p):
            rows[i][4] = repr(float(value))
    template = ROOT / 'admet_official/submission_template.csv'
    raw = template.read_bytes()
    spans = field_spans(raw)
    pieces, cursor = [], 0
    for i, row in enumerate(rows[1:], 1):
        lo, hi = spans[i][4]
        pieces.extend((raw[cursor:lo], row[4].encode('ascii')))
        cursor = hi
    pieces.append(raw[cursor:])
    payload = b''.join(pieces)
    folder = ROOT / 'admet_submissions/v179_graph_embedding_boosting'
    folder.mkdir(exist_ok=True)
    file = folder / '我从山北来带着兰花草-小分子药物预测技术创新赛-20261027.csv'
    if file.exists():
        assert file.read_bytes() == payload
    else:
        file.write_bytes(payload)
    assert table(file) == rows
    fmt = verify(file, template)
    changes = Counter()
    for a, z in zip(original[1:], rows[1:]):
        if a[0] not in selected:
            assert a == z
        if a[4] != z[4]:
            changes[a[0]] += 1
    assert set(changes) == set(selected) and 'hia_hou' not in changes
    outspans = field_spans(payload)
    pieces, cursor = [], 0
    for i in range(1, len(rows)):
        lo, hi = outspans[i][4]
        a, z = spans[i][4]
        pieces.extend((payload[cursor:lo], raw[a:z]))
        cursor = hi
    pieces.append(payload[cursor:])
    assert b''.join(pieces) == raw
    assert fmt['rows'] == 16379 and fmt['tasks'] == 22 and fmt['utf8_bom'] and fmt['crlf_only']
    fmt.update(nonprediction_bytes_exact=True, original_drug_id_strings_exact=True,
               changed_tasks=dict(changes), unchanged_tasks=22 - len(changes),
               independent_final_fusion_replay=True)
    report = dict(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM', version='v179', training_version='v178',
                  baseline='v145', baseline_official_score=51.166, baseline_sha256=v.ANCHOR_SHA,
                  file=file.relative_to(ROOT).as_posix(), sha256=b.sha(file), selected=selected,
                  checks=checks, verification=fmt, official_score=None, automatically_uploaded=False,
                  test_labels_used=False, new_external_data=False,
                  limitations='Previously reused public labels and approximate upstream controls. No official improvement guarantee.')
    b.save(OUT / 'evidence.json', report)
    b.save(OUT / 'independent_verification.json', dict(status='PASS', checks=checks))
    b.save(folder / 'model_evidence.json', report)
    b.save(file.with_suffix('.verification.json'), fmt)
    print(json.dumps(dict(status=report['status'], changed_tasks=dict(changes), sha256=report['sha256'])), flush=True)


if __name__ == '__main__':
    main()
