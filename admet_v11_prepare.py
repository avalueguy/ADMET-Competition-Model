"""Reconstruct and check the scored task baseline without changing historical CSVs."""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    protocol=json.loads(Path('admet_v11_protocol.json').read_text(encoding='utf-8'))
    root=Path('admet_release_v10_guard')
    upload=next(Path('admet_submissions/v10_guard').glob('*.csv'))
    assert sha(upload)==protocol['baseline_sha256']
    scored=read(upload);assert len(scored)==16379
    assert set(r['task'] for r in scored)==set(protocol['tasks'])
    checks=[]
    for task in protocol['tasks']:
        version=7 if task=='dili' else 10
        source=Path(f'admet_release_v{version}')/task
        dest=root/task;dest.mkdir(exist_ok=True)
        files=[f'submission_{task}.csv','validation_predictions.csv']
        for name in files:
            shutil.copy2(source/name,dest/name)
            assert (source/name).read_bytes()==(dest/name).read_bytes()
        rows=read(dest/files[0]);slice_=[r for r in scored if r['task']==task]
        assert len(rows)==len(slice_)
        for i,(a,b) in enumerate(zip(rows,slice_)):
            assert int(b['sample_idx'])==i and a['Drug_ID']==b['drug_id']
            assert math.isclose(float(a['Y']),float(b['y_pred']),rel_tol=1e-12,abs_tol=1e-12)
        record=dict(task=task,accepted=False,reason='frozen scored v10_guard task baseline',
            source_release=f'v{version}',source_selection_sha256=sha(source/'selection.json'),
            submission_sha256=sha(dest/files[0]),validation_sha256=sha(dest/files[1]),rows=len(rows))
        (dest/'selection.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        checks.append(record)
    manifest=dict(status='VERIFIED_TASK_BASELINE',submission_sha256=sha(upload),
        platform_score_user_reported=49.375594,score_file_linkage='inferred from conversation',
        tasks=checks,historical_submissions_unchanged=True)
    (root/'baseline_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'],tasks=len(checks),rows=len(scored))))


if __name__=='__main__':main()
