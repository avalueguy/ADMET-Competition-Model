"""Independent replicate noise weight and model reload and accepted-task assembly."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v264_replicate_noise_gp as model
import v148_experiment_runner as runner
from finalize_v124_local import rows as table
from repack_upload_template import field_spans
from verify_submission import verify
v, b, r83 = model.v, model.b, model.r83
OUT = ROOT / 'admet_model_outputs_v265'


def main():
    OUT.mkdir(exist_ok=True)
    cp = model.OUT / 'confirmation.json'
    for _ in range(720):
        if cp.exists() and json.loads(cp.read_text())['status'] == 'COMPLETE':
            break
        time.sleep(15)
    else:
        raise TimeoutError('v264 incomplete; preserve partial results')
    conf = json.loads(cp.read_text())
    frozen = json.loads((model.OUT / 'frozen_selection.json').read_text())
    assert conf['signature'] == frozen['signature'] == model.sig()
    selected = {r['task']: r for r in conf['records'] if r['accepted']}
    if not selected:
        b.save(OUT / 'no_candidate.json', dict(status='NO_CONFIRMED_NEW_MODEL',
               training_version='v264', baseline='v145', official_baseline_score=51.166,
               submission_created=False, confirmation_sha256=b.sha(cp)))
        print('NO_CONFIRMED_NEW_MODEL', flush=True)
        return
    d = model.load()
    original = table(v.ANCHOR)
    rows = [r.copy() for r in original]
    checks = []
    for task, record in selected.items():
        k = d['tasks'].index(task)
        choice = next(r for r in frozen['selected'] if r['task'] == task)
        assert runner.confirm(d, k, model.predict, choice) == record
        for part in v.PARTS:
            a = model.design(d, k, part, record['config'])
            folder = model.OUT / 'models' / task / part / model.key(record['config'])
            cached = np.load(folder / 'predictions.npz')
            assert str(cached['signature']) == model.sig()
            for name in ('fit', 'query'):
                np.testing.assert_array_equal(a['state'][name], cached[name])
            labels = d['y'][a['state']['query']].copy()
            try:
                d['y'][a['state']['query']] = 999999.
                d.pop('_v96_assays', None)
                z = model.design(d, k, part, record['config'])
                from v264_replicate_noise_gp import same
                same(a, z)
            finally:
                d['y'][a['state']['query']] = labels
                d.pop('_v96_assays', None)
            p, hashes = model.replay(d, k, part, record['config'])
            hashes['transform.joblib'] = b.sha(folder / 'transform.joblib')
            np.testing.assert_array_equal(p, cached['p'])
            candidate = v.mix(record['metric'], v.baseline(d, k, part), p, record['weight'])
            if part == 'final':
                positions = [i for i, row in enumerate(original) if i and row[0] == task]
                assert len(positions) == len(candidate)
                for i, value in zip(positions, candidate):
                    rows[i][4] = repr(float(value))
            else:
                expected = next(r for r in record['confirmation'] if r['part'] == part)
                np.testing.assert_allclose(r83.score(record['metric'], d['y'][a['state']['query']], candidate), expected['candidate'], atol=1e-12, rtol=0)
            checks.append(dict(task=task, part=part, model_reload=True,
                               query_label_perturbation_invariant=True,
                               model_hashes=hashes))
            print(json.dumps(dict(stage='independent_replay', task=task, part=part)), flush=True)
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
    folder = ROOT / 'admet_submissions/v265_replicate_noise_gp'
    folder.mkdir(exist_ok=True)
    file = folder / '我从山北来带着兰花草-小分子药物预测技术创新赛-20261029.csv'
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
    report = dict(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM', version='v265', training_version='v264',
                  baseline='v145', baseline_official_score=51.166, baseline_sha256=v.ANCHOR_SHA,
                  file=file.relative_to(ROOT).as_posix(), sha256=b.sha(file), selected=selected,
                  checks=checks, 
                   verification=fmt, official_score=None, automatically_uploaded=False,
                  test_labels_used=False, new_external_data=False,
                  limitations='Reused public labels and approximate upstream controls. No official improvement guarantee.')
    b.save(OUT / 'evidence.json', report)
    b.save(OUT / 'independent_verification.json', dict(status='PASS', checks=checks))
    b.save(folder / 'model_evidence.json', report)
    b.save(file.with_suffix('.verification.json'), fmt)
    print(json.dumps(dict(status=report['status'], changed_tasks=dict(changes), sha256=report['sha256'])), flush=True)


if __name__ == '__main__':
    main()
