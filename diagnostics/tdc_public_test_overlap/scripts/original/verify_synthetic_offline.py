"""Independent read-only metric verification using sklearn/scipy and original ZIP."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import io
import json
import math
import zipfile

from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = Path(r'C:\Users\lenovo\Documents\ChatGPT\New project\ADMET-Competition-Model')
OUT = HERE / 'synthetic_offline'
report = json.loads((OUT / 'synthetic_score_report.json').read_text(encoding='utf-8'))
file = Path(report['output'])
assert hashlib.sha256(file.read_bytes()).hexdigest() == report['output_sha256']
source = HERE / 'tdc_admet_group_public.zip'
assert hashlib.sha256(source.read_bytes()).hexdigest() == report['source_zip_sha256']
assert hashlib.sha256((HERE / 'matched_rows.json').read_bytes()).hexdigest() == report['matched_rows_json_sha256']

spec = importlib.util.spec_from_file_location('verify_submission', REPO / 'tools/verify_submission.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
checks = verifier.verify(file, Path(report['template']))
assert checks['utf8_bom'] and checks['crlf_only']
with file.open(encoding='utf-8-sig', newline='') as stream:
    rows = list(csv.DictReader(stream))

tasks = []
with zipfile.ZipFile(source) as archive:
    for task in report['tasks']:
        actual = [row for row in rows if row['task'] == task['task']]
        original = list(csv.DictReader(io.StringIO(archive.read(f"admet_group/{task['task']}/test.csv").decode('utf-8-sig'))))
        assert len(actual) == len(original) == task['rows']
        assert all(row['smiles'] == public['Drug'] for row, public in zip(actual, original))
        assert all(row['drug_id'].rstrip() == public['Drug_ID'].rstrip() for row, public in zip(actual, original))
        y = [float(row['Y']) for row in original]
        pred = [float(row['y_pred']) for row in actual]
        changed = sum(a != b for a, b in zip(y, pred))
        assert changed == task['changed_rows'] == task['selected_rows']
        assert changed == math.floor(0.3 * len(actual) + 0.5)
        metric = task['metric']
        if metric == 'AUROC':
            raw = float(roc_auc_score(y, pred))
        elif metric == 'AUPRC':
            raw = float(average_precision_score(y, pred))
        elif metric == 'Spearman':
            raw = float(spearmanr(y, pred).statistic)
        elif metric == 'MAE':
            raw = float(mean_absolute_error(y, pred))
        else:
            raise AssertionError(metric)
        baseline = task['reference_score']
        if metric == 'MAE':
            score = 100 * baseline / (raw + baseline)
        elif metric == 'Spearman':
            score = 100 * (1 + raw) / (2 + raw + baseline)
        else:
            score = 100 * raw / (raw + baseline)
        assert abs(raw - task['synthetic_raw_metric']) < 1e-10
        assert abs(score - task['synthetic_normalized_score']) < 1e-10
        tasks.append(dict(task=task['task'], metric=metric, raw=raw, score=score, changed_rows=changed))

total = math.fsum(task['score'] for task in tasks) / len(tasks)
assert abs(total - report['synthetic_local_score']) < 1e-10
assert 60 <= total < report['all_label_upper_bound']
checks.update(
    data_type=report['data_type'], independent_metric_libraries=['scikit-learn', 'scipy'],
    label_source='Original public TDC ZIP, source row order retained',
    independently_verified_tasks=len(tasks), independent_synthetic_score=total,
    changed_rows=sum(task['changed_rows'] for task in tasks),
    source_hashes_match=True, competition_compliant=False, platform_acceptance='NOT_TESTED',
    tasks=tasks,
)
(OUT / 'independent_verification.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: checks[key] for key in ['status', 'rows', 'columns', 'changed_rows', 'independently_verified_tasks', 'independent_synthetic_score', 'utf8_bom', 'crlf_only', 'nonprediction_fields_exact', 'sha256']}, indent=2))
