"""Read-only prediction checks and local v20 bookkeeping; no model fitting."""
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'), newline='')))


def percent_rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start+end)/2/len(values)
        for k in range(start,end):
            out[order[k]] = rank
        start = end
    return out


def main():
    old = next((ROOT/'admet_submissions/v19').glob('*.csv'))
    new = next((ROOT/'admet_submissions/v20').glob('*.csv'))
    template = ROOT/'admet_official/submission_template.csv'
    proof = read(new.parent/'verification.json')
    assert digest(old) == '6d8b151afb359f35e6e19c16e537f6824db2862a159851ea13d86325d64e8cfb'
    assert digest(new) == proof['sha256']
    a, b, t = rows(new), rows(old), rows(template)
    assert len(a) == len(b) == len(t) == 16380
    assert a[0] == b[0] == t[0] == ['task','sample_idx','drug_id','smiles','y_pred','metric']
    blob = new.read_bytes()
    assert blob.startswith(b'\xef\xbb\xbf')
    assert blob.count(b'\r\n') == blob.count(b'\n') == 16380
    changed, total = Counter(), Counter()
    seen = set()
    task_rows = {}
    for r, prior, official in zip(a[1:], b[1:], t[1:]):
        assert len(r) == 6 and r[:4]+r[5:] == prior[:4]+prior[5:] == official[:4]+official[5:]
        key = (r[0], r[1])
        assert key not in seen
        seen.add(key)
        value = float(r[4])
        assert math.isfinite(value)
        if r[5] in {'AUROC','AUPRC'}:
            assert 0 <= value <= 1
        total[r[0]] += 1
        task_rows.setdefault(r[0], []).append((float(prior[4]), value))
        if r[4] != prior[4]:
            changed[r[0]] += 1
    assert len(total) == 22 and set(changed) == set(proof['changed'])
    checks = []
    for task in changed:
        evidence = ROOT/'admet_model_outputs_v20_rare'/task
        decision = read(evidence/'decision.json')['selection']
        audit = read(evidence/'audit.json')
        checkpoint = read(evidence/'checkpoint.json')
        confirmation = read(evidence/'confirmation.json')
        assert decision['eligible'] and decision['weight'] > 0
        assert decision['oof_gain'] > 0 and decision['valid_gain'] > 0
        assert audit['alignment_passed'] and audit['scaffold_overlap'] == 0
        assert checkpoint['model_replay_passed'] and checkpoint['train_plus_valid_only']
        assert checkpoint['prediction_sha256'] == digest(evidence/'rare_test.csv')
        release = ROOT/'admet_release_v20_rare'/task/f'submission_{task}.csv'
        preds = [float(r[0]) for r in rows(release)[1:]]
        actual = [r[1] for r in task_rows[task]]
        assert len(actual) == len(preds) and all(abs(x-y)<1e-12 for x,y in zip(actual,preds))
        candidate = [float(r[0]) for r in rows(evidence/'rare_test.csv')[1:]]
        baseline = [r[0] for r in task_rows[task]]
        if decision['mode'] == 'rank':
            candidate, baseline = percent_rank(candidate), percent_rank(baseline)
        weight = decision['weight']
        expected = [(1-weight)*x+weight*y for x,y in zip(baseline,candidate)]
        assert len(expected) == len(actual) and all(abs(x-y)<1e-12 for x,y in zip(actual,expected))
        values = task_rows[task]
        inversions = sum((a0-b0)*(a1-b1)<0 for i,(a0,a1) in enumerate(values) for b0,b1 in values[i+1:])
        checks.append(dict(task=task, changed_rows=changed[task], strict_order_reversals=inversions,
                           oof_gain=decision['oof_gain'], valid_gain=decision['valid_gain'],
                           repeat_gain=confirmation['gain'],
                           valid_bootstrap_95=audit['valid_uncertainty']['percentile_95']))
    report = dict(status='VERIFIED_LOCAL', submission=str(new.resolve()), sha256=digest(new),
                  baseline='v19', baseline_score=49.993751, platform_score=None,
                  bytes=len(blob), rows=16379, columns=6, tasks=22,
                  encoding='UTF-8 BOM', line_endings='CRLF', template_metadata_exact=True,
                  changed_rows=sum(changed.values()), changed_tasks=dict(changed),
                  unchanged_tasks=sorted(set(total)-set(changed)), prediction_replay_verified=True,
                  checks=checks, local_gpu_training=False, automatically_uploaded=False,
                  limitation='Small gains on repeatedly reused splits; confidence intervals include zero. Experimental candidate, not a guaranteed score increase.')
    (new.parent/'local_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    current = read(ROOT/'admet_current_candidate.json')
    current.update(recommended_release='v19', experimental_release='v20', experimental_submission=str(new.resolve()),
                   experimental_sha256=digest(new), candidate_platform_score=None,
                   platform_anchor_release='v19', platform_anchor_score=49.993751,
                   awaiting='Optional platform evaluation of v20 experimental candidate',
                   report='admet_diagnostics/v20_delivery_report.md',
                   verification='admet_submissions/v20/local_verification.json',
                   target_achieved=False, next_version_minimum_target=50.593751)
    (ROOT/'admet_current_candidate.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
