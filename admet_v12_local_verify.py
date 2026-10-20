"""Retrieve only lightweight v12 outputs and verify exact template and frozen v11 rows."""
import csv
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    workspace = Path.cwd().resolve()
    with tarfile.open('admet_v12_review.tar.gz', 'r:gz') as archive:
        for member in archive.getmembers():
            p = Path(member.name)
            if not member.isfile() or not p.parts:
                raise ValueError(member.name)
            allowed = p.parts[0] in {'admet_model_outputs_v12', 'admet_release_v12'}
            allowed |= len(p.parts)>1 and p.parts[0]=='admet_submissions' and p.parts[1]=='v12'
            allowed |= p.parts[0] in {'admet_diagnostics','logs'} and p.name.startswith('v12')
            if not allowed or p.suffix not in {'.csv','.json','.log','.md'}:
                raise ValueError(member.name)
            target = (workspace/p).resolve()
            if not target.is_relative_to(workspace):
                raise ValueError(member.name)
            if p.parts[0] == 'logs':
                target = workspace/'admet_diagnostics/v12_remote_logs'/p.name
            content = archive.extractfile(member).read()
            if p.parts[0] == 'admet_submissions' and target.exists() and target.read_bytes() != content:
                raise ValueError('different existing v12 submission')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    root = Path('admet_release_v12')
    verification = json.loads((root/'independent_verification.json').read_text(encoding='utf-8'))
    if verification['status'] == 'NO_PROMOTION':
        print(json.dumps(verification)); return
    assert verification['status']=='VERIFIED' and verification['independent_replay']
    upload = Path(verification['submission']); blob = upload.read_bytes()
    assert digest(upload) == verification['sha256']
    assert blob.startswith(b'\xef\xbb\xbf') and blob.count(b'\r\n') == 16380
    assert blob.count(b'task,sample_idx,drug_id,smiles,y_pred,metric') == 1
    rows = list(csv.reader(io.StringIO(blob.decode('utf-8-sig'), newline='')))
    with open('admet_official/submission_template.csv', encoding='utf-8-sig', newline='') as handle:
        template = list(csv.reader(handle))
    original = Path('admet_submissions/v11/我从山北来带着兰花草-小分子药物预测技术创新赛-20260925.csv')
    assert digest(original) == '082183f0c462081ea8735a41ec2e47326ef4cdcaf9d1188ae86a006f8057122d'
    with open(original, encoding='utf-8-sig', newline='') as handle:
        old = list(csv.reader(handle))
    assert len(rows)==len(template)==len(old)==16380 and rows[0]==template[0]==old[0]
    changed = set(); unchanged = set(); accepted = set(verification['accepted'])
    for row, ref, prior in zip(rows[1:], template[1:], old[1:]):
        assert len(row)==6 and row[:4]==ref[:4]==prior[:4] and row[5:]==ref[5:]==prior[5:]
        value = float(row[4]); assert math.isfinite(value)
        if row[5] in ['AUROC','AUPRC']:
            assert 0<=value<=1
        if row[0] not in accepted:
            assert row==prior; unchanged.add(row[0])
        elif row[4] != prior[4]:
            changed.add(row[0])
    assert changed == accepted and len(changed|unchanged)==22
    result = dict(status='VERIFIED_LOCAL_AND_REMOTE', submission=str(upload.resolve()), sha256=digest(upload),
        rows=16379, columns=6, tasks=22, changed=sorted(changed), unchanged=sorted(unchanged),
        rejected_task_prediction_strings_unchanged=True, encoding='UTF-8 BOM', line_endings='CRLF',
        baseline_platform_score=49.381036, platform_anchor_version='v11', platform_anchor_score=49.381036, new_platform_score=None, automatic_submission=False)
    (root/'local_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__=='__main__':
    main()
