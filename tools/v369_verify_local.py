"""Receive remote evidence and independently check the delivered submission."""
from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import csv
import hashlib
import json
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/transformer_stage4_v369'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def receive():
    receipt = read(EXP / 'result_archive.json')
    archive = ROOT / 'releases/v369_remote_results.tar.gz'
    assert receipt['status'] == 'PASS' and sha(archive) == receipt['sha256']
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        names = [m.name for m in members]
        assert len(set(names)) == len(names)
        for member in members:
            assert member.isfile(), member.name
            target = (EXP / member.name).resolve()
            assert target.is_relative_to(EXP.resolve()), member.name
            content = tf.extractfile(member).read()
            if target.exists():
                assert target.read_bytes() == content, member.name
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    manifest = read(EXP / 'result_manifest.json')
    assert manifest['signature'] == receipt['signature']
    for entry in manifest['files']:
        assert sha(EXP / entry['path']) == entry['sha256'], entry['path']
    for name in ('v364_graph_transformer.py', 'v369_graph_transformer.py',
                 'v369_run_stage4.py', 'v369_verify_remote.py', 'v369_collect_remote.py'):
        assert sha(ROOT / 'tools' / name) == sha(EXP / name), name
    result = dict(status='PASS', archive_sha256=sha(archive),
                  archive_members=len(members), verified_files=len(manifest['files']),
                  signature=manifest['signature'], source_files_match=True)
    save(ROOT / 'docs/v369_remote_receipt_verification.json', result)
    print(json.dumps(result))


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def delivery():
    result = read(ROOT / 'docs/v369_stage4_delivery.json')
    if not result.get('csv'):
        assert result['status'] == 'NO_ELIGIBLE_CANDIDATE'
        print(json.dumps(dict(status='PASS', no_candidate=True)))
        return
    path = ROOT / result['csv']
    assert sha(path) == result['csv_sha256']
    old = table(EXP / 'v364_baseline.csv')
    new = table(path)
    template = table(EXP / 'submission_template.csv')
    assert len(old) == len(new) == len(template) == 16379
    assert len(set((r['task'], r['sample_idx']) for r in new)) == 16379
    assert len(set(r['task'] for r in new)) == 22
    z = np.load(EXP / 'official_rows.npz')
    manifest = read(EXP / 'data_manifest.json')
    updated = {r['task']: r for r in result['fusion']}
    changed = Counter()
    for a, b, c in zip(old, new, template):
        assert {k: v for k, v in b.items() if k != 'y_pred'} == {k: v for k, v in c.items() if k != 'y_pred'}
        if a != b:
            changed[b['task']] += 1
            assert b['task'] in updated
    assert dict(changed) == result['changed_rows']
    endpoint_checks = []
    for task, spec in updated.items():
        baseline = [r for r in old if r['task'] == task]
        candidate = [r for r in new if r['task'] == task]
        control = np.load(EXP / 'controls' / task / 'final.npz')
        expert = np.load(EXP / 'outputs/tasks' / task / 'final' / spec['mode'] / 'predictions.npz')
        query = control['query']
        assert np.isnan(z['y'][query]).all()
        assert np.all(z['split'][query] == 2)
        assert np.all(z['tid'][query] == manifest['tasks'].index(task))
        assert [r['smiles'] for r in candidate] == z['smiles'][z['mi'][query]].tolist()
        before = np.array([float(r['y_pred']) for r in baseline])
        after = np.array([float(r['y_pred']) for r in candidate])
        from scipy.stats import rankdata
        expected = (1-spec['weight'])*rankdata(before)/len(before) + spec['weight']*rankdata(expert['p'])/len(before)
        np.testing.assert_array_equal(after, expected)
        assert np.isfinite(after).all()
        if task == 'ppbr_az':
            assert np.all((after >= 0) & (after <= 100))
        endpoint_checks.append(dict(task=task, rows=len(after),
                                    actual_smiles_order_verified=True,
                                    fusion_error=0.0))
    release = read(ROOT / 'docs/v369_release_archive.json')
    archive = ROOT / release['path']
    assert sha(archive) == release['sha256']
    release_manifest = path.parent / 'release_manifest.json'
    entries = read(release_manifest)['files']
    with tarfile.open(archive) as tf:
        assert tf.extractfile(release_manifest.relative_to(ROOT).as_posix()).read() == release_manifest.read_bytes()
        for entry in entries:
            assert sha(ROOT / entry['path']) == entry['sha256'], entry['path']
            assert hashlib.sha256(tf.extractfile(entry['path']).read()).hexdigest() == entry['sha256'], entry['path']
    report = dict(status='PASS', signature=result['signature'],
                  csv_sha256=sha(path), rows=16379, tasks=22,
                  changed_rows=dict(changed), endpoints=endpoint_checks,
                  archive_current_bytes_and_members_verified=True,
                  verified_release_files=len(entries), official_score=None)
    save(ROOT / 'docs/v369_local_delivery_audit.json', report)
    progress = read(ROOT / 'docs/v369_stage4_progress.json')
    progress.update(status='COMPLETE', completed_at_hong_kong=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
                    new_submission_delivered=True, submission=result['csv'],
                    submission_sha256=sha(path), official_score=None,
                    known_best_unchanged='v364:51.269', local_delivery_audit='PASS')
    save(ROOT / 'docs/v369_stage4_progress.json', progress)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['receive', 'delivery'])
    args = parser.parse_args()
    (receive if args.stage == 'receive' else delivery)()
