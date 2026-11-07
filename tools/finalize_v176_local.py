"""Recompute saved v175 fusion locally before exposing any v176 upload file."""
import math
import shutil
from collections import Counter
from pathlib import Path

from finalize_v124_local import ROOT, array, load, ranks, rows, save, sha
from repack_upload_template import field_spans
from verify_submission import verify


def main():
    folder = ROOT / 'admet_model_outputs_v176'
    report = load(folder / 'evidence.json')
    audit = load(folder / 'independent_verification.json')
    backup = load(ROOT / 'releases/v176_shared_graph_batch_20261027.archive.json')
    archive_check = load(ROOT / 'docs/v176_local_archive_verification.json')
    plan = load(ROOT / 'docs/v175_training_plan.json')
    frozen = load(ROOT / 'admet_model_outputs_v175/frozen_selection.json')
    confirmation = load(ROOT / 'admet_model_outputs_v175/confirmation.json')
    assert audit['status'] == archive_check['status'] == 'PASS'
    assert archive_check['all_member_hashes_verified'] and archive_check['dependency_hashes_verified']
    assert archive_check['archive_sha256'] == backup['sha256']
    assert sha(ROOT / 'tools/v175_shared_graph_hybrid.py') == plan['source_sha256']
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
        assert check['global_scaffold_exclusion']
        model_folder = ROOT / 'admet_model_outputs_v175/models' / check['part'] / 'h96_l3_s5000'
        for name, expected in check['model_hashes'].items():
            assert sha(model_folder / name) == expected

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
    saved = ROOT / 'admet_model_outputs_v175/models/final/h96_l3_s5000/predictions.npz'
    for task, record in selected.items():
        k = tasks.index(task)
        p = array(saved, 'p' + str(k))
        assert array(saved, 'query' + str(k)) == queries[task]
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
    save(ROOT / 'docs/v176_local_submission_verification.json', fmt)
    dest = Path('D:/bili_download/v176_shared_graph_hybrid')
    dest.mkdir(parents=True, exist_ok=True)
    upload = dest / file.name
    if upload.exists():
        assert upload.read_bytes() == raw
    else:
        shutil.copy2(file, upload)
    assert sha(upload) == report['sha256']
    save(upload.with_suffix('.verification.json'), fmt)
    delivery = dict(status='LOCAL_DELIVERY_COMPLETE_READY_FOR_EDGE', version='v176', training_version='v175',
                    baseline='v145', baseline_official_score=51.166, csv=report['file'],
                    upload_copy=str(upload), sha256=report['sha256'], changed_tasks=dict(changes),
                    unchanged_tasks=22 - len(changes), official_score=None, target_achieved=False,
                    target_score=52, automatically_uploaded=False, verification=fmt, archive=backup)
    save(ROOT / 'docs/v176_local_delivery.json', delivery)
    save(dest / 'v176_delivery.json', delivery)
    state = load(ROOT / 'CURRENT_STATUS.json')
    state.update(active_experiment='v176', active_experiment_status=delivery['status'],
                 recommended_submission_version='v176', new_submission_created=True,
                 running_new_training_processes=0)
    state['v176_result'] = delivery
    state['releases']['176'] = delivery
    save(ROOT / 'CURRENT_STATUS.json', state)
    print(__import__('json').dumps(delivery, ensure_ascii=True))


if __name__ == '__main__':
    main()
