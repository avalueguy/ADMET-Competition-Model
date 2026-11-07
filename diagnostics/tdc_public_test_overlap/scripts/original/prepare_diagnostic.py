"""Align the public TDC test observations for a disclosed overlap diagnostic.

This does not train a model. Public test labels must never be represented as
held-out model predictions or imported into the existing training experiments.
"""
from pathlib import Path
import collections
import csv
import hashlib
import io
import json
import math
import zipfile

HERE = Path(__file__).resolve().parent
REPO = Path(r'C:\Users\lenovo\Documents\ChatGPT\New project\ADMET-Competition-Model')
TEMPLATE = REPO / 'admet_official/submission_template.csv'
SOURCE = HERE / 'tdc_admet_group_public.zip'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    with TEMPLATE.open(encoding='utf-8-sig', newline='') as handle:
        template = list(csv.DictReader(handle))
    rows, tasks = [], []
    with zipfile.ZipFile(SOURCE) as archive:
        for task in dict.fromkeys(row['task'] for row in template):
            target = [row for row in template if row['task'] == task]
            member = f'admet_group/{task}/test.csv'
            content = archive.read(member)
            public = list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
            assert len(public) == len(target)
            pools = collections.defaultdict(set)
            for record in public:
                pools[(record['Drug_ID'], record['Drug'])].add(record['Y'])
            differences, conflicts = [], 0
            for index, (record, source) in enumerate(zip(target, public)):
                assert record['sample_idx'] == str(index)
                assert record['smiles'] == source['Drug']
                assert math.isfinite(float(source['Y']))
                if record['drug_id'] != source['Drug_ID']:
                    # The only differences in the successful template are names'
                    # trailing spaces. Do not silently authorize other mismatches.
                    assert record['drug_id'].rstrip() == source['Drug_ID'].rstrip()
                    differences.append(dict(sample_idx=index,
                        template_id=record['drug_id'], tdc_id=source['Drug_ID']))
                conflicts += len(pools[(source['Drug_ID'], source['Drug'])]) > 1
                rows.append(dict(**record, public_label=source['Y']))
            tasks.append(dict(task=task, rows=len(target), exact_ordered_smiles=len(target),
                exact_ids=len(target)-len(differences), id_differences=differences,
                duplicate_key_conflicting_measurement_rows=conflicts,
                source_member=member,
                source_member_sha256=hashlib.sha256(content).hexdigest()))
    assert len(rows) == 16379 and len(tasks) == 22
    report = dict(status='PASS', rows=len(rows), task_count=len(tasks), tasks=tasks,
        matched_by='task + sample_idx; exact ordered SMILES for every row, original repeated-measurement order retained; five trailing-space name differences recorded',
        template=str(TEMPLATE), template_sha256=sha(TEMPLATE),
        source_url='https://dataverse.harvard.edu/api/access/datafile/4426004?format=original',
        source_zip_sha256=sha(SOURCE),
        data_type='PUBLIC_TEST_OBSERVATION_LOOKUP_NOT_MODEL_PREDICTIONS',
        competition_rule='Saved competition rules prohibit external data and require model-generated predictions. This export is a disclosed label-overlap diagnostic, not a compliant model submission.',
        no_training_performed=True, automatic_platform_submission=False)
    (HERE/'overlap_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (HERE/'matched_rows.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(status='PASS', rows=len(rows), tasks=len(tasks),
        id_whitespace_differences=sum(len(t['id_differences']) for t in tasks))))


if __name__ == '__main__':
    main()
