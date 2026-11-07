"""Read-only cross-check of disclosed public labels and synthetic scoring fixture."""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import zipfile

from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

BASE = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((BASE / 'synthetic_offline/synthetic_score_report.json').read_text(encoding='utf-8'))
    source = BASE / 'source/tdc_admet_group_public.zip'
    synthetic = BASE / 'synthetic_offline/OFFLINE_SYNTHETIC_seed20261106.csv'
    labels = BASE / 'public_labels/PUBLIC_TEST_LABELS_NOT_MODEL.csv'
    assert sha(source) == report['source_zip_sha256']
    assert sha(synthetic) == report['output_sha256']
    assert sha(BASE / 'source/submission_template.csv') == report['template_sha256']
    assert sha(BASE / 'source/reference_scores_v387.json') == report['reference_scores_file_sha256']
    assert sha(labels) == 'c9f4839000828c35a0456387dc896d8ea4d4583d15cd748f72390adeb410b58f'
    actual, public, template = rows(synthetic), rows(labels), rows(BASE / 'source/submission_template.csv')
    assert len(actual) == len(public) == len(template) == 16379
    for path in (synthetic, labels):
        raw = path.read_bytes()
        assert raw.startswith(b'\xef\xbb\xbf') and b'\n' not in raw.replace(b'\r\n', b'')
    for fixture, answer, expected in zip(actual, public, template):
        assert fixture.keys() == answer.keys() == expected.keys()
        assert all(fixture[key] == answer[key] == expected[key] for key in expected if key != 'y_pred')
        assert math.isfinite(float(fixture['y_pred']))
        if fixture['metric'] in ('AUROC', 'AUPRC'):
            assert 0 <= float(fixture['y_pred']) <= 1
    scores, changed = [], 0
    with zipfile.ZipFile(source) as archive:
        for task in report['tasks']:
            predicted_rows = [row for row in actual if row['task'] == task['task']]
            answer_rows = [row for row in public if row['task'] == task['task']]
            original = list(csv.DictReader(io.StringIO(archive.read(f"admet_group/{task['task']}/test.csv").decode('utf-8-sig'))))
            assert len(original) == len(predicted_rows) == task['rows']
            assert all(row['smiles'] == src['Drug'] for row, src in zip(predicted_rows, original))
            truth = [float(row['Y']) for row in original]
            assert [float(row['y_pred']) for row in answer_rows] == truth
            predicted = [float(row['y_pred']) for row in predicted_rows]
            count = sum(a != b for a, b in zip(truth, predicted))
            assert count == task['changed_rows'] == math.floor(0.3 * len(original) + 0.5)
            changed += count
            metric = task['metric']
            raw = {
                'AUROC': lambda: roc_auc_score(truth, predicted),
                'AUPRC': lambda: average_precision_score(truth, predicted),
                'MAE': lambda: mean_absolute_error(truth, predicted),
                'Spearman': lambda: spearmanr(truth, predicted).statistic,
            }[metric]()
            reference = task['reference_score']
            score = (100 * reference / (raw + reference) if metric == 'MAE'
                     else 100 * (1 + raw) / (2 + raw + reference) if metric == 'Spearman'
                     else 100 * raw / (raw + reference))
            assert abs(float(raw) - task['synthetic_raw_metric']) < 1e-10
            assert abs(float(score) - task['synthetic_normalized_score']) < 1e-10
            scores.append(float(score))
    total = math.fsum(scores) / 22
    assert changed == report['changed_rows'] == 4917
    assert abs(total - report['synthetic_local_score']) < 1e-10
    print(json.dumps(dict(status='PASS', rows=16379, tasks=22, changed_rows=changed,
        synthetic_local_score=total, competition_compliant=False, model_performance=False), indent=2))


if __name__ == '__main__':
    main()
