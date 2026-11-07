"""Prepare immutable official-only inputs; no training on the local computer."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/transformer_stage3_v368'
TASKS = ('ames', 'clearance_microsome_az', 'half_life_obach', 'vdss_lombardo')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    assert not (OUT / 'data_manifest.json').exists(), 'Preserve prepared experiment inputs'
    OUT.mkdir(parents=True, exist_ok=True)
    previous = ROOT / 'experiments/transformer_stage1_v364'
    old = json.loads((previous / 'data_manifest.json').read_text(encoding='utf-8'))
    best = json.loads((ROOT / 'docs/current_best_after_v364_20261104.json').read_text(encoding='utf-8'))
    baseline = ROOT / best['csv']
    assert sha(baseline) == best['csv_sha256']
    with baseline.open(encoding='utf-8-sig', newline='') as stream:
        current = list(csv.DictReader(stream))
    raw = previous / 'official_rows.npz'
    if 'official_rows_sha256' in old:
        assert sha(raw) == old['official_rows_sha256']
    shutil.copyfile(raw, OUT / raw.name)
    d = dict(np.load(raw, allow_pickle=False))
    assert np.isnan(d['y'][d['split'] == 2]).all()
    groups = d['groups'][d['mi']]
    canonical = d['canonical'][d['mi']]
    records = []
    sources = []
    for task in TASKS:
        for part in ('fold0', 'fold1', 'fold2', 'valid', 'final'):
            source = ROOT / 'admet_model_outputs_v146/controls' / task / (part + '.npz')
            control = dict(np.load(source))
            fit, query, pred = control['fit'], control['query'], control['p'].copy()
            assert np.isfinite(d['y'][fit]).all() and np.isfinite(pred).all()
            assert not np.intersect1d(fit, query).size and np.all(np.diff(query) > 0)
            assert np.all(d['tid'][fit] == old['tasks'].index(task))
            assert np.all(d['tid'][query] == old['tasks'].index(task))
            assert np.all(d['split'][fit] < (2 if part == 'final' else 1))
            if part != 'final':
                assert not np.intersect1d(groups[fit], groups[query]).size
                assert not np.intersect1d(canonical[fit], canonical[query]).size
            else:
                final = np.array([float(row['y_pred']) for row in current if row['task'] == task])
                np.testing.assert_array_equal(final, pred)
            target = OUT / 'controls' / task / (part + '.npz')
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, fit=fit, query=query, p=pred)
            sources.append(dict(path=source.relative_to(ROOT).as_posix(), sha256=sha(source)))
            records.append(dict(task=task, part=part, fit=len(fit), query=len(query),
                                file=target.relative_to(OUT).as_posix(), sha256=sha(target),
                                upstream_OOF='partial historical proxy'))
    shutil.copyfile(baseline, OUT / 'v364_baseline.csv')
    shutil.copyfile(previous / 'submission_template.csv', OUT / 'submission_template.csv')
    manifest = dict(tasks=old['tasks'], metrics=old['metrics'], pilot_tasks=TASKS,
                    rows=old['rows'], unique_exact_smiles=old['unique_exact_smiles'], baseline=best,
                    official_rows_sha256=sha(raw), source_inputs=old['source_inputs'], sources=sources,
                    controls=records, test_labels_parsed=False, all_training_remote_only=True,
                    host='connect.nmb2.seetacloud.com', port=25870)
    save(OUT / 'data_manifest.json', manifest)
    names = ('v364_graph_transformer.py', 'v368_graph_transformer.py', 'v368_run_stage3.py',
             'v368_verify_remote.py', 'v368_collect_remote.py')
    for name in names:
        shutil.copyfile(ROOT / 'tools' / name, OUT / name)
    archive = ROOT / 'releases/v368_stage3_inputs.tar.gz'
    with tarfile.open(archive, 'w:gz') as tf:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                tf.add(path, arcname=path.relative_to(OUT).as_posix())
    report = dict(status='PASS', archive=archive.relative_to(ROOT).as_posix(),
                  sha256=sha(archive), bytes=archive.stat().st_size, tasks=TASKS, controls=records)
    save(ROOT / 'docs/v368_input_package.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
