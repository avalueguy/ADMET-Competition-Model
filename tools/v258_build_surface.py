"""Construct frozen exposure features from official structures, without label use."""
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v256_exposed_polarity as model


def one(item):
    mi, smiles = item
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        record = model.geometry.exposure(smiles)
    except Exception as exc:
        record = dict(status='FAILED_FEATURE', error=type(exc).__name__ + ': ' + str(exc))
    return mi, model.feature_vector(record), record


def original_structures():
    # Only strings are used. Never parse a Y column or open valid/test labels.
    template = model.b.table(ROOT / 'admet_official/submission_template.csv')
    tasks = list(dict.fromkeys(x['task'] for x in template))
    all_smiles, lookup, selected = [], {}, {}
    for task in tasks:
        for part in ('train', 'valid', 'test'):
            rows = ([dict(Drug=x['smiles']) for x in template if x['task'] == task]
                    if part == 'test' else model.b.table(ROOT / 'admet_competition_data/data_benchmark' / task / (part + '.csv')))
            for row in rows:
                sm = row['Drug']
                if sm not in lookup:
                    lookup[sm] = len(all_smiles)
                    all_smiles.append(sm)
                if task in model.TASKS:
                    selected[lookup[sm]] = sm
    assert len(all_smiles) == 46202
    return selected


def main():
    frozen = json.loads((ROOT / 'docs/v256_frozen_sources.json').read_text())
    for row in frozen['files']:
        assert model.b.sha(ROOT / row['path']) == row['sha256']
    folder = model.OUT / 'features'
    assert not (folder / 'build_start.json').exists(), 'Inspect previous build; no automatic restart'
    assert not (folder / 'audit.json').exists()
    folder.mkdir(parents=True, exist_ok=True)
    source = original_structures()
    digest = hashlib.sha256(json.dumps(sorted(source.items()), separators=(',', ':')).encode()).hexdigest()
    model.b.save(folder / 'build_start.json', dict(signature=model.sig(), molecules=len(source),
                 smiles_sha256=digest, assay_labels_used=False, workers=6))
    chunks = folder / 'chunks'
    chunks.mkdir(exist_ok=True)
    ordered = sorted(source.items())
    x, records = {}, {}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=6) as pool:
        for lo in range(0, len(ordered), 96):
            result = list(pool.map(one, ordered[lo:lo + 96], chunksize=1))
            for mi, vec, record in result:
                x[mi], records[mi] = vec, record
            joblib.dump(result, chunks / (str(lo) + '.joblib'), compress=3)
            report = dict(status='BUILDING', signature=model.sig(), done=len(x), total=len(source),
                          seconds=time.monotonic() - started, outcomes=dict(Counter(r['status'] for r in records.values())), workers=6)
            model.b.save(folder / 'build_progress.json', report)
            print(json.dumps(report), flush=True)
    mi = np.array(sorted(x))
    matrix = np.stack([x[int(i)] for i in mi])
    np.savez_compressed(folder / 'surface.npz', mi=mi, x=matrix, signature=model.sig())
    joblib.dump(records, folder / 'records.joblib', compress=3)
    # Feature viability is a label-free precondition, not a predictive score.
    coverage = {}
    template = model.b.table(ROOT / 'admet_official/submission_template.csv')
    by_sm = {sm: i for i, sm in source.items()}
    for task in model.TASKS:
        rows = model.b.table(ROOT / 'admet_competition_data/data_benchmark' / task / 'train.csv')
        ids = [by_sm[r['Drug']] for r in rows]
        coverage[task] = sum(records[i]['status'] == 'PASS' for i in ids) / len(ids)
        assert coverage[task] >= .90, (task, coverage[task])
    audit = dict(status='PASS', signature=model.sig(), smiles_sha256=digest,
                 features_sha256=model.b.sha(folder / 'surface.npz'),
                 records_sha256=model.b.sha(folder / 'records.joblib'),
                 molecules=len(source), dimensions=len(model.FEATURE_NAMES),
                 feature_names=list(model.FEATURE_NAMES), outcomes=dict(Counter(r['status'] for r in records.values())),
                 train_structure_coverage=coverage, assay_labels_used=False,
                 seconds=time.monotonic() - started, workers=6)
    model.b.save(folder / 'audit.json', audit)
    model.b.save(folder / 'build_progress.json', dict(status='COMPLETE', **{k: v for k, v in audit.items() if k != 'status'}))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        model.b.save(model.OUT / 'feature_build_failure.json', dict(status='FAILED', traceback=traceback.format_exc(), retry_permitted=False))
        raise
