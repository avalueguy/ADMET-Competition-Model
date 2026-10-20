"""Independently replay exported source fusion and verify the official CSV schema."""
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from admet_v24_local_verify import ranks
ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'), newline='')))


def main():
    out = ROOT / 'admet_model_outputs_v42'
    dest = ROOT / 'admet_submissions/v42'
    file = next(dest.glob('*.csv'))
    baseline = next((ROOT / 'admet_submissions/v39').glob('*.csv'))
    info = json.loads((dest / 'verification.json').read_text(encoding='utf-8'))
    report = json.loads((out / 'analysis.json').read_text(encoding='utf-8'))
    replay = json.loads((out / 'model_replay.json').read_text(encoding='utf-8'))
    compact = json.loads((out / 'test_predictions_compact.json').read_text(encoding='utf-8'))
    manifest = json.loads((out / 'source_manifest.json').read_text(encoding='utf-8'))
    assert info['signature'] == report['signature'] == replay['signature']
    assert replay['status'] == 'SOURCE_MODEL_REPLAYS_VERIFIED'
    for name, digest in manifest.items():
        assert sha(ROOT / name) == digest
    verified = set()
    for version, source in replay['sources'].items():
        assert source['status'] in ['FINAL_NEURAL_AND_ISOLATION_REPLAY_PASSED', 'FINAL_PARENT_AND_ISOLATION_REPLAY_PASSED']
        for row in source['tasks']:
            assert row['model_reload_passed']
            assert row['hidden_target_labels_invariant'] and row['target_auxiliary_block_invariant']
            verified.add((row['task'], int(version)))
    assert sha(file) == info['sha256'] and sha(baseline) == info['baseline_sha256'] and sha(file) != sha(baseline)
    data = file.read_bytes()
    assert data.startswith(b'\xef\xbb\xbf') and data.count(b'\r\n') == data.count(b'\n') == 16380
    now, prior, template = read(file), read(baseline), read(ROOT / 'admet_official/submission_template.csv')
    assert len(now) == len(prior) == len(template) == 16380
    assert now[0] == prior[0] == template[0] == ['task', 'sample_idx', 'drug_id', 'smiles', 'y_pred', 'metric']
    changed = Counter()
    bytask = defaultdict(list)
    keys = set()
    for row, old, target in zip(now[1:], prior[1:], template[1:]):
        assert len(row) == 6 and row[:4]+row[5:] == old[:4]+old[5:] == target[:4]+target[5:]
        key = (row[0], row[1])
        assert key not in keys
        keys.add(key)
        value = float(row[4])
        assert math.isfinite(value)
        if row[5] in ['AUROC', 'AUPRC']:
            assert 0 <= value <= 1
        if value != float(old[4]):
            changed[row[0]] += 1
        bytask[row[0]].append((float(old[4]), value))
    assert len(bytask) == 22 and dict(changed) == info['changed_tasks']
    rank_changes = {}
    for r in report['tasks']:
        task = r['task']
        s = r['selection']
        old = [a for a, _ in bytask[task]]
        actual = [a for _, a in bytask[task]]
        if s is None:
            assert old == actual
            continue
        parts = compact[task]
        assert len(parts) == len(s['sources'])
        candidate = [0.] * len(old)
        for part, source in zip(parts, s['sources']):
            assert {key: part[key] for key in ['version', 'variant', 'weight']} == source
            assert (task, source['version']) in verified
            values = part['predictions']
            assert len(values) == len(old) and all(math.isfinite(v) for v in values)
            if s['name'] == 'complement_equal' and r['metric'] != 'MAE':
                values = ranks(values)
            candidate = [a+source['weight']*b for a, b in zip(candidate, values)]
        if r['metric'] != 'MAE':
            rank_changes[task] = sum(a != b for a, b in zip(ranks(old), ranks(actual)))
            old, candidate = ranks(old), ranks(candidate)
        expected = [(1-s['weight'])*a+s['weight']*b for a, b in zip(old, candidate)]
        assert all(abs(a-b) < 1e-12 for a, b in zip(actual, expected))
    result = dict(status='INDEPENDENT_SOURCE_FUSION_AND_CSV_PASSED', sha256=sha(file), rows=16379, tasks=22,
                  columns=6, bytes=len(data), changed_tasks=dict(changed), bom=True, crlf=True, template_identity_exact=True,
                  source_hashes_match=True, final_fusion_replayed=True, remote_model_reload=True,
                  label_isolation_passed=True, ranking_changed_rows=rank_changes, local_gpu_training=False)
    (dest / 'local_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
