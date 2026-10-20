"""Fill and validate the exact official template without reordering samples."""
from __future__ import annotations
import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np
import pandas as pd


def same_id(a, b):
    if str(a) == str(b):
        return True
    try:
        return Decimal(str(a)) == Decimal(str(b))
    except InvalidOperation:
        return False


def build(source, output, template_path=Path('admet_official/submission_template.csv')):
    template = pd.read_csv(template_path, dtype={'drug_id': str}, keep_default_na=False)
    result = template.copy()
    result['y_pred'] = np.nan
    checks = []
    for task, part in template.groupby('task', sort=False):
        test = pd.read_csv(Path('admet_competition_data/data_benchmark')/task/'test.csv', dtype={'Drug_ID': str}, keep_default_na=False)
        pfile = source/task/f'submission_{task}.csv'
        pred = pd.read_csv(pfile, dtype={'Drug_ID': str}, keep_default_na=False)
        assert len(part) == len(test) == len(pred), task
        assert part.sample_idx.astype(int).tolist() == list(range(len(test))), task
        assert part.smiles.tolist() == test.Drug.tolist(), task
        assert all(same_id(a,b) for a,b in zip(part.drug_id, test.Drug_ID)), task
        assert all(same_id(a,b) for a,b in zip(pred.Drug_ID, test.Drug_ID)), task
        y = pd.to_numeric(pred.Y, errors='raise').to_numpy(float)
        assert np.isfinite(y).all(), task
        if part.metric.iloc[0] in ['AUROC','AUPRC']:
            assert ((y >= 0) & (y <= 1)).all(), task
        result.loc[part.index, 'y_pred'] = y
        checks.append(dict(task=task, rows=len(y), source=str(pfile),
                           sha256=hashlib.sha256(pfile.read_bytes()).hexdigest()))
    assert len(result) == 16379
    assert result.drop(columns='y_pred').equals(template.drop(columns='y_pred'))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    reread = pd.read_csv(output)
    assert np.isfinite(reread.y_pred.to_numpy(float)).all()
    output.with_suffix('.audit.json').write_text(json.dumps(dict(rows=len(result),
        columns=list(result.columns), template_sha256=hashlib.sha256(template_path.read_bytes()).hexdigest(),
        submission_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), tasks=checks),indent=2),encoding='utf8')
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('admet_model_outputs_v3'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(build(a.source,a.output))
