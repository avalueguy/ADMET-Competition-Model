"""Verify the downloaded v74 archive, replay its CSV, and record local delivery.

This performs no model training and requires only NumPy and the standard library.
The CUDA model reload audit is produced separately on the authorized server.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from verify_submission import read, verify
from repack_upload_template import field_spans

PREFIX = 'v74_v73_20261021'
OUT = ROOT / 'admet_model_outputs_v74'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def check_archive():
    meta = load(ROOT / 'releases' / (PREFIX + '.archive.json'))
    archive = ROOT / meta['path']
    assert meta['status'] == 'PASS' and sha(archive) == meta['sha256']
    manifest_path = ROOT / 'releases' / (PREFIX + '.manifest.json')
    manifest = load(manifest_path)
    expected = {r['path']: r for r in manifest['files']}
    manifest_name = manifest_path.relative_to(ROOT).as_posix()
    seen = set()
    extracted = 0
    # A single streaming pass also verifies the weights retained only in the tar.
    with tarfile.open(archive, 'r|gz') as tf:
        for member in tf:
            rel = PurePosixPath(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            assert member.name not in seen
            seen.add(member.name)
            raw = tf.extractfile(member).read()
            if member.name == manifest_name:
                assert raw == manifest_path.read_bytes()
                continue
            r = expected[member.name]
            assert len(raw) == r['bytes'] and hashlib.sha256(raw).hexdigest() == r['sha256'], member.name
            target = (ROOT / member.name).resolve()
            assert target.is_relative_to(ROOT.resolve())
            selected = member.name.startswith(('admet_model_outputs_v74/', 'admet_submissions/v74_v73/')) or member.name.startswith('v74_') and member.name.endswith('.log')
            if member.name == 'admet_model_outputs_v74/pipeline_status.json':
                # Its archived value predates archive completion; retain live final status.
                selected = False
            if selected:
                if target.exists():
                    assert target.read_bytes() == raw, f'Existing active artifact differs: {member.name}'
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                    extracted += 1
            elif member.name.endswith('.py') or member.name == 'admet_v74_protocol.json':
                assert target.is_file() and sha(target) == r['sha256'], f'Local source differs: {member.name}'
    assert seen == set(expected) | {manifest_name}
    assert len(seen) == meta['members']
    print(json.dumps(dict(stage='LOCAL_ARCHIVE_PASS',members=len(seen),extracted=extracted)), flush=True)
    return meta


def main():
    archive_meta = check_archive()
    release = load(OUT / 'release.json')
    model = load(OUT / 'model_verification.json')
    confirmation = load(OUT / 'confirmation.json')
    protocol = load(ROOT / 'admet_v74_protocol.json')
    assert release['status'] == 'VERIFIED_CANDIDATE_AWAITING_PLATFORM'
    assert model['status'] == 'PASS' and model['independent_process']
    assert sha(OUT / 'model_verification.json') == release['verification_sha256']
    assert release['signature'] == model['signature'] == confirmation['signature']
    assert sorted(release['selected']) == sorted(confirmation['accepted'])
    assert len(model['checks']) == model['models']
    for record in model['checks']:
        assert sha(ROOT / record['path']) == record['sha256']
    candidate = ROOT / release['file']
    baseline = next((ROOT / 'admet_submissions/v73_v72').glob('*.csv'))
    template = ROOT / 'admet_official/submission_template.csv'
    assert sha(template) == '8ee41589d2c98e7b93081c058b3f928ac09bea4b3ffc899fa13677051e9baa20'
    assert sha(candidate) == release['sha256']
    assert sha(baseline) == release['anchor_sha256'] == protocol['anchor_sha256']
    audit = verify(candidate, template)
    assert audit['rows'] == 16379 and audit['tasks'] == 22 and audit['utf8_bom'] and audit['crlf_only']
    rows, old = read(candidate), read(baseline)
    replay = load(OUT / 'fusion_replay.json')
    changed = {}
    for task in dict.fromkeys(r[0] for r in rows[1:]):
        actual = [r for r in rows[1:] if r[0] == task]
        prior = [r for r in old[1:] if r[0] == task]
        if task not in release['selected']:
            assert actual == prior
            continue
        choice = release['selected'][task]['choice']
        saved = []
        for seed in protocol['final_seeds']:
            with np.load(OUT / task / 'final' / str(seed) / 'predictions.npz') as z:
                assert str(z['signature']) == release['signature']
                saved.append(z['p'])
        expert = np.mean(saved, axis=0)[choice['heads']].mean(0)
        with np.load(OUT / task / 'final/tree/predictions.npz') as z:
            expert = (1 - choice['tree_weight']) * expert + choice['tree_weight'] * z['p']
        anchor = np.array([float(r[4]) for r in prior])
        expected = (1 - choice['anchor_weight']) * anchor + choice['anchor_weight'] * expert
        np.testing.assert_array_equal(expected, np.array([float(r[4]) for r in actual]))
        np.testing.assert_array_equal(expert, replay[task]['expert'])
        np.testing.assert_array_equal(anchor, replay[task]['anchor'])
        np.testing.assert_array_equal(expected, replay[task]['result'])
        changed[task] = sum(r[4] != s[4] for r, s in zip(actual, prior))
    assert changed == model['changed_tasks'] and any(changed.values())
    raw, ref = candidate.read_bytes(), template.read_bytes()
    s, t = field_spans(raw), field_spans(ref)
    chunks, cursor = [], 0
    for i in range(1, len(rows)):
        lo, hi = s[i][4]
        a, z = t[i][4]
        chunks.extend((raw[cursor:lo], ref[a:z]))
        cursor = hi
    chunks.append(raw[cursor:])
    assert b''.join(chunks) == ref
    # Preserve the remote audit bytes, which are also archived and hash checked.
    audit.update(independent_local_fusion_replay='PASS', nonprediction_bytes_exact=True,
                 unchanged_tasks_prediction_strings_exact=True, changed_tasks=changed,
                 independent_model_verification=f"PASS_{model['models']}_MODELS_{model['final_models']}_FINAL")
    delivery_dir = Path('D:/bili_download/v74_v73')
    delivery_dir.mkdir(parents=True, exist_ok=True)
    delivery = delivery_dir / candidate.name
    if delivery.exists():
        assert delivery.read_bytes() == raw
    else:
        shutil.copy2(candidate, delivery)
    save(delivery.with_suffix('.verification.json'), audit)
    result = dict(status='PASS_LOCAL_DELIVERY_AWAITING_PLATFORM', candidate=release['file'],
        upload_copy=str(delivery), sha256=sha(candidate), rows=16379, columns=6, tasks=22,
        template_sha256=sha(template), format_verification='PASS',
        independent_model_verification=audit['independent_model_verification'],
        independent_local_fusion_replay='PASS', nonprediction_bytes_exact=True,
        changed_tasks=changed, unchanged_tasks=22-len(changed),
        archive=archive_meta, all_archive_member_hashes_verified=True,
        official_score=None, anchor='v73', anchor_official_score_user_reported=50.640,
        confirmation=confirmation['tasks'], exact_full_v73_oof_comparison=False,
        confirmation_is_unseen_external_data=False,
        test_labels_used=False, new_branch_pretrained_weights=False,
        anchor_contains_historical_pretraining=True, automatic_platform_submission=False)
    save(ROOT / 'docs/v74_delivery.json', result)
    state = load(ROOT / 'CURRENT_STATUS.json')
    state.update(latest_completed_model_branch='v74', latest_completed_model_branch_is_proven_best=False,
        rollback_baseline='v73', official_score=None, best_official_version_user_reported='v73',
        best_official_score_user_reported=50.640, target_achieved=False,
        priority_next_step='Submit verified v74 CSV manually and record platform score; keep v73 as best user-reported baseline.',
        running_new_iteration_processes_at_completion=0)
    state['active_iteration'].update(status='LOCAL_DELIVERY_COMPLETE_AWAITING_PLATFORM',
        changed_tasks=changed, rejected_tasks=[r['task'] for r in confirmation['tasks'] if not r['accepted']],
        independent_model_reload='PASS', local_archive_verified=True, running_training_processes=0)
    state['releases']['74'] = dict(csv=release['file'],sha256=sha(candidate),
        status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',baseline='v73',official_score=None,
        changed_tasks=changed,unchanged_tasks=22-len(changed),
        independent_model_reload=result['independent_model_verification'],
        local_archive=archive_meta,desktop_upload_copy=str(delivery),
        baseline_contains_historical_pretraining=True)
    save(ROOT / 'CURRENT_STATUS.json', state)
    print(json.dumps({k:v for k,v in result.items() if k!='confirmation'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
