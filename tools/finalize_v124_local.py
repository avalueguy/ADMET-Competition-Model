"""Verify final fusion without ML dependencies and deliver the exact CSV."""
import ast
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from verify_submission import verify
from repack_upload_template import field_spans


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(path, key):
    # Read only one-dimensional numeric .npy arrays, never pickle objects.
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(key + '.npy')
    assert raw[:6] == b'\x93NUMPY'
    major = raw[6]
    assert major in (1, 2, 3)
    width = 2 if major == 1 else 4
    size = int.from_bytes(raw[8:8+width], 'little')
    end = 8 + width + size
    header = ast.literal_eval(raw[8+width:end].decode('utf-8'))
    assert len(header['shape']) == 1 and not header['fortran_order']
    dtype = header['descr']
    assert dtype in ('<f8', '<f4', '<i8', '<i4'), dtype
    code = {'<f8': 'd', '<f4': 'f', '<i8': 'q', '<i4': 'i'}[dtype]
    return list(struct.unpack('<' + str(header['shape'][0]) + code, raw[end:]))


def ranks(values):
    result = [0.] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + 1 + stop) / 2
        for position in order[start:stop]:
            result[position] = average
        start = stop
    return result


def rows(path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'), newline='')))


def main():
    report = load(ROOT / 'admet_model_outputs_v124/evidence.json')
    audit = load(ROOT / 'admet_model_outputs_v119/independent_verification.json')
    backup = load(ROOT / 'releases/v124_models_history_20261027.archive.json')
    archive_check = load(ROOT / 'docs/v124_local_archive_verification.json')
    assert audit['status'] == 'PASS' and audit['models'] == 10
    assert archive_check['all_member_hashes_verified'] and archive_check['dependency_hashes_verified']
    assert archive_check['archive_sha256'] == backup['sha256']
    file = ROOT / report['file']
    assert sha(file) == report['sha256']
    baseline = next((ROOT / 'admet_submissions/v114_herg_history52').glob('*.csv'))
    assert sha(baseline) == report['baseline_sha256']
    current, actual = rows(baseline), rows(file)
    history = next((ROOT / 'admet_submissions/v112_history_expansion').glob('*.csv'))
    historical = rows(history)
    assert sha(history) == report['sources']['ames']['source_sha256']
    task = 'half_life_obach'
    positions = [i for i, row in enumerate(current) if i and row[0] == task]
    saved = ROOT / 'admet_model_outputs_v119/models/final/full/predictions.npz'
    expert = array(saved, 'p')
    query = array(saved, 'query')
    old_query = array(ROOT / 'admet_model_outputs_v68/half_life_obach/final/predictions.npz', 'query')
    assert query == old_query and len(query) == len(positions) == 135
    old = [float(current[i][4]) for i in positions]
    a, z = ranks(old), ranks(expert)
    replay = [.5*x/len(old) + .5*y/len(old) for x, y in zip(a, z)]
    assert all(abs(float(actual[i][4]) - value) < 1e-15 for i, value in zip(positions, replay))
    changed = Counter()
    for i, (oldrow, newrow) in enumerate(zip(current[1:], actual[1:]), 1):
        assert oldrow[:4] + oldrow[5:] == newrow[:4] + newrow[5:]
        if newrow[0] == 'ames':
            assert newrow == historical[i]
        elif newrow[0] != task:
            assert oldrow == newrow
        if oldrow[4] != newrow[4]:
            changed[newrow[0]] += 1
    assert dict(changed) == report['verification']['changed_tasks']
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
    fmt.update(nonprediction_bytes_exact=True, independent_final_fusion_replay=True)
    save(file.with_suffix('.verification.json'), fmt)
    dest = Path('D:/bili_download/v124_matched_models_history')
    dest.mkdir(parents=True, exist_ok=True)
    upload = dest / file.name
    if upload.exists():
        assert upload.read_bytes() == raw
    else:
        shutil.copy2(file, upload)
    assert sha(upload) == report['sha256']
    shutil.copy2(file.with_suffix('.verification.json'), upload.with_suffix('.verification.json'))
    shutil.copy2(ROOT / 'docs/ITERATION_V124_20261027.md', dest / 'v124_说明.md')
    delivery = dict(status='LOCAL_DELIVERY_COMPLETE_AWAITING_PLATFORM', version='v124',
        baseline='v114', baseline_official_score=51.091, csv=report['file'], upload_copy=str(upload),
        sha256=report['sha256'], changed_tasks=dict(changed), changed_predictions=sum(changed.values()),
        unchanged_tasks=20, rows=16379, columns=6, tasks=22, nonprediction_bytes_exact=True,
        independent_final_fusion_replay=True, independent_model_reload='PASS_10_CHECKPOINTS',
        archive=backup, all_archive_member_hashes_verified=True, dependencies_verified=True,
        official_score=None, target_score=52, target_achieved=False, automatically_uploaded=False,
        github_push=False, server_shutdown=False)
    save(ROOT / 'docs/v124_local_delivery.json', delivery)
    save(dest / 'v124_delivery.json', delivery)
    state_path = ROOT / 'CURRENT_STATUS.json'
    state = load(state_path)
    state.update(handoff_date='2026-10-27', last_updated_hong_kong='2026-10-27',
        latest_completed_model_branch='v124', latest_completed_model_branch_is_proven_best=False,
        rollback_baseline='v114', official_score=51.091, target_achieved=False,
        best_official_version='v114', best_official_version_user_reported='v114',
        best_official_score_user_reported=51.091, last_user_reported_official_version='v114',
        last_user_reported_official_score=51.091, per_task_platform_scores_available=True,
        active_experiment='v124', active_experiment_status=delivery['status'],
        recommended_submission_version='v124', new_submission_created=True,
        running_new_training_processes=0, running_new_iteration_processes_at_completion=0,
        priority_next_step='Measure v124 official task results; continue historical recovery and matched-baseline model development for weak tasks.')
    state['active_iteration'] = dict(version='v124', baseline='v114', highest_official_version='v114',
        highest_official_score=51.091, status=delivery['status'], csv=delivery['csv'],
        running_training_processes=0, official_score=None, github_push=False, server_shutdown_requested=False)
    state['releases']['114'] = dict(official_score=51.091, status='USER_REPORTED_OFFICIAL',
        csv=str(baseline.relative_to(ROOT)), sha256=sha(baseline), scorecard='docs/platform_scorecard_v114_20261027.json')
    state['releases']['124'] = delivery
    state['v124_result'] = delivery
    for number in (115,116,117,121,123):
        conf = load(ROOT / f'admet_model_outputs_v{number}/confirmation.json')
        assert conf['status'] == 'COMPLETE' and conf['accepted'] == []
        state[f'v{number}_experiment_result'] = dict(status='COMPLETE', accepted=[], official_score=None)
    state['v119_experiment_result'] = dict(status='COMPLETE', accepted=['half_life_obach'], official_score=None)
    save(state_path, state)
    readme = ROOT / 'README.md'
    text = readme.read_text(encoding='utf-8-sig')
    heading = '# 当前官方最佳 v114（51.091）；v124 新模型候选待评分\n\n'
    heading += 'v124 合入新训练的双目标半衰期 CatBoost 专家与历史实测更好的 AMES 预测，更新 1,591 条预测，其余 20 个任务保留 v114。本轮另测试序列核、完整结构条件树、分布加权和两种排序路线；确认未稳定胜出的模型没有合入。\n\n'
    heading += '新半衰期模型按实际 v68 基线的相同目标训练行比较，四组 Spearman 增益为 +0.026607、+0.062370、+0.057918、+0.011460。10 个模型检查点独立重载、最终融合、本地 953 个归档文件和模板非预测字节检查通过。valid 只有 67 条，区间跨零，不能保证官方涨分或达到 52。\n\n'
    heading += '[实验记录](docs/ITERATION_V124_20261027.md) · [本地交付](docs/v124_local_delivery.json) · [提交 CSV](' + report['file'] + ')\n\n'
    heading += '以下是保留的历史记录，当前状态以本段和 CURRENT_STATUS.json 为准。\n\n'
    if not text.startswith(heading):
        readme.write_text(heading + text, encoding='utf-8')
    print(json.dumps(delivery, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
