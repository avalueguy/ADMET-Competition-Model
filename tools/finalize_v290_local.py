"""Independently recompute frozen v289 stereochemical context repair fusion before exposing v290."""
import math
import shutil
from collections import Counter
from pathlib import Path

from finalize_v124_local import ROOT, array, load, ranks, rows, save, sha
from repack_upload_template import field_spans
from verify_submission import verify


def main():
    folder = ROOT / 'admet_model_outputs_v290'
    report = load(folder / 'evidence.json')
    audit = load(folder / 'independent_verification.json')
    backup = load(ROOT / 'releases/v290_stereo_schema_fix_20261031.archive.json')
    archive_check = load(ROOT / 'docs/v290_local_archive_verification.json')
    plan = load(ROOT / 'docs/v289_training_plan.json')
    frozen = load(ROOT / 'admet_model_outputs_v289/frozen_selection.json')
    confirmation = load(ROOT / 'admet_model_outputs_v289/confirmation.json')
    assert audit['status'] == archive_check['status'] == 'PASS'
    assert archive_check['all_member_hashes_verified'] and archive_check['dependency_hashes_verified']
    assert archive_check['archive_sha256'] == backup['sha256']
    assert sha(ROOT / 'tools/v289_stereo_schema_fix.py') == plan['source_sha256']
    assert confirmation['status'] == 'COMPLETE'
    assert confirmation['signature'] == frozen['signature']
    selected = report['selected']
    assert selected and set(selected) == set(confirmation['accepted'])
    assert 'hia_hou' not in selected
    assert set(selected) == {r['task'] for r in confirmation['records'] if r['accepted']}
    for task, record in selected.items():
        assert record == next(r for r in confirmation['records'] if r['task'] == task)
        choice = next(r for r in frozen['selected'] if r['task'] == task)
        assert all(record[key] == choice[key] for key in choice)
        assert all(r['gain'] > 0 for r in record['confirmation'])
        minimum = .012 if record['metric'] == 'MAE' else .008 if record['metric'] == 'Spearman' else .003
        assert sum(r['gain'] for r in record['confirmation']) / 4 > minimum
        assert {r['part'] for r in record['bootstrap']} == {'fold2', 'valid'}
        assert all(r['lower_positive'] and r['gain_ci90'][0] > 0 for r in record['bootstrap'])
    checks = report['checks']
    assert checks == audit['checks']
    assert len(checks) == 5 * len(selected)
    assert {(r['task'], r['part']) for r in checks} == {
        (t, p) for t in selected for p in ('fold0', 'fold1', 'fold2', 'valid', 'final')}
    for check in checks:
        assert check['model_reload'] and check['query_label_perturbation_invariant']
        model_folder = ROOT / 'admet_model_outputs_v289/models' / check['task'] / check['part'] / 'v145_stereo_specific_context_fixed_schema'
        for name, expected in check['model_hashes'].items():
            assert sha(model_folder / name) == expected
        for path, expected in check['original_model_hashes'].items():
            assert sha(ROOT / path) == expected

    baseline = next((ROOT / 'admet_submissions/v145_feedback_repair').glob('*.csv'))
    file = ROOT / report['file']
    assert sha(baseline) == report['baseline_sha256'] == '3ffd68785b3fc321c8e81a32b23a09cf2493ff26240555a3472a315c572706e7'
    assert sha(file) == report['sha256']
    old, actual = rows(baseline), rows(file)
    tasks = list(dict.fromkeys(r[0] for r in old[1:]))
    assert len(tasks) == 22
    cursor, queries = 0, {}
    for task in tasks:
        for part in ('train', 'valid'):
            source = ROOT / 'admet_competition_data/data_benchmark' / task / (part + '.csv')
            cursor += len(rows(source)) - 1
        count = sum(r[0] == task for r in old[1:])
        queries[task] = list(range(cursor, cursor + count))
        cursor += count
    assert cursor == 81809

    for task, record in selected.items():
        saved = ROOT / 'admet_model_outputs_v289/models' / task / 'final' / 'v145_stereo_specific_context_fixed_schema' / 'predictions.npz'
        p = array(saved, 'p')
        assert array(saved, 'query') == queries[task]
        positions = [i for i, r in enumerate(old) if i and r[0] == task]
        a = [float(old[i][4]) for i in positions]
        assert len(a) == len(p)
        weight = record['weight']
        if record['metric'] == 'MAE':
            replay = [(1 - weight) * x + weight * z for x, z in zip(a, p)]
        else:
            replay = [(1 - weight) * x / len(a) + weight * z / len(a)
                      for x, z in zip(ranks(a), ranks(p))]
        assert all(math.isfinite(z) for z in replay)
        assert all(abs(float(actual[i][4]) - z) < 1e-14 for i, z in zip(positions, replay))

    changes = Counter()
    assert len(old) == len(actual) == 16380
    for a, z in zip(old[1:], actual[1:]):
        assert a[:4] + a[5:] == z[:4] + z[5:]
        if a[0] not in selected:
            assert a == z
        if a[4] != z[4]:
            changes[a[0]] += 1
    assert dict(changes) == report['verification']['changed_tasks']
    assert set(changes) == set(selected)
    template = ROOT / 'admet_official/submission_template.csv'
    fmt = verify(file, template)
    raw, ref = file.read_bytes(), template.read_bytes()
    ss, tt = field_spans(raw), field_spans(ref)
    pieces, cursor = [], 0
    for i in range(1, len(actual)):
        lo, hi = ss[i][4]
        first, last = tt[i][4]
        pieces.extend((raw[cursor:lo], ref[first:last]))
        cursor = hi
    pieces.append(raw[cursor:])
    assert b''.join(pieces) == ref
    assert fmt['rows'] == 16379 and fmt['tasks'] == 22 and fmt['utf8_bom'] and fmt['crlf_only']
    fmt.update(nonprediction_bytes_exact=True, original_drug_id_strings_exact=True,
               independent_final_fusion_replay=True, unchanged_task_predictions_exact=True,
               fixed_confirmation_gates_rechecked=True, archived_model_hashes_verified=True)
    save(ROOT / 'docs/v290_local_submission_verification.json', fmt)
    dest = Path('D:/bili_download/v290_stereo_schema_fix')
    dest.mkdir(parents=True, exist_ok=True)
    upload = dest / file.name
    if upload.exists():
        assert upload.read_bytes() == raw
    else:
        shutil.copy2(file, upload)
    assert sha(upload) == report['sha256']
    save(upload.with_suffix('.verification.json'), fmt)
    delivery = dict(status='LOCAL_DELIVERY_COMPLETE_MANUAL_SUBMISSION_ONLY', version='v290', training_version='v289',
                    baseline='v145', baseline_official_score=51.166, csv=report['file'],
                    upload_copy=str(upload), sha256=report['sha256'], changed_tasks=dict(changes),
                    unchanged_tasks=22 - len(changes), official_score=None, target_achieved=False,
                    target_score=52, automatically_uploaded=False, verification=fmt, archive=backup)
    save(ROOT / 'docs/v290_local_delivery.json', delivery)
    save(dest / 'v290_delivery.json', delivery)
    state = load(ROOT / 'CURRENT_STATUS.json')
    state.update(active_experiment='v290', active_experiment_status=delivery['status'],
                 recommended_submission_version='v290', new_submission_created=True,
                 running_new_training_processes=0)
    state['v290_result'] = delivery
    state['releases']['285'] = delivery
    save(ROOT / 'CURRENT_STATUS.json', state)
    print(__import__('json').dumps(delivery, ensure_ascii=True))


if __name__ == '__main__':
    main()
