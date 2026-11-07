"""Prepare five numeric endpoints with all adopted v364 control layers replayed."""
from pathlib import Path
import csv, hashlib, json, shutil, tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/transformer_stage5_v370'
TASKS = ('caco2_wang', 'ld50_zhu', 'lipophilicity_astrazeneca', 'ppbr_az', 'solubility_aqsoldb')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    assert not (OUT / 'data_manifest.json').exists(), 'Preserve frozen inputs'
    OUT.mkdir(parents=True, exist_ok=True)
    previous = ROOT / 'experiments/transformer_stage1_v364'
    old = json.loads((previous / 'data_manifest.json').read_text(encoding='utf-8'))
    best = json.loads((ROOT / 'docs/current_best_after_v364_20261104.json').read_text(encoding='utf-8'))
    baseline = ROOT / best['csv']
    assert sha(baseline) == best['csv_sha256']
    with baseline.open(encoding='utf-8-sig', newline='') as f:
        current = list(csv.DictReader(f))
    raw = previous / 'official_rows.npz'
    shutil.copyfile(raw, OUT / raw.name)
    d = dict(np.load(raw))
    assert np.isnan(d['y'][d['split'] == 2]).all()
    groups, canonical = d['groups'][d['mi']], d['canonical'][d['mi']]
    records, sources = [], []
    for task in TASKS:
        for part in ('fold0', 'fold1', 'fold2', 'valid', 'final'):
            if task in ('lipophilicity_astrazeneca', 'solubility_aqsoldb'):
                source = previous / 'controls' / task / (part + '.npz')
            else:
                source = ROOT / 'admet_model_outputs_v146/controls' / task / (part + '.npz')
            a = np.load(source)
            fit, query, pred = a['fit'], a['query'], a['p'].copy()
            sources.append(source)
            if part != 'final' and task == 'ppbr_az':
                paths = list((ROOT / 'admet_model_outputs_v304/models' / task / part).glob('*/predictions.npz'))
                assert len(paths) == 1
                expert = np.load(paths[0])
                np.testing.assert_array_equal(expert['query'], query)
                pred = .55 * pred + .45 * expert['p']
                sources.append(paths[0])
            if part != 'final' and task in ('lipophilicity_astrazeneca', 'solubility_aqsoldb'):
                path = previous / 'outputs/tasks' / task / part / 'family_transfer/predictions.npz'
                expert = np.load(path)
                np.testing.assert_array_equal(expert['query'], query)
                np.testing.assert_array_equal(expert['fit'], fit)
                pred = .8 * pred + .2 * expert['p']
                sources.append(path)
            assert not np.intersect1d(fit, query).size and np.all(np.diff(query) > 0)
            assert np.all(d['tid'][fit] == old['tasks'].index(task))
            assert np.all(d['tid'][query] == old['tasks'].index(task))
            assert np.isfinite(d['y'][fit]).all() and np.isfinite(pred).all()
            assert np.all(d['split'][fit] < (2 if part == 'final' else 1))
            if part != 'final':
                assert not np.intersect1d(groups[fit], groups[query]).size
                assert not np.intersect1d(canonical[fit], canonical[query]).size
            else:
                rows = [r for r in current if r['task'] == task]
                assert [r['smiles'] for r in rows] == d['smiles'][d['mi'][query]].tolist()
                pred = np.array([float(r['y_pred']) for r in rows])
            dest = OUT / 'controls' / task / (part + '.npz')
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, fit=fit, query=query, p=pred)
            records.append(dict(task=task, part=part, fit=len(fit), query=len(query),
                file=dest.relative_to(OUT).as_posix(), sha256=sha(dest),
                upstream_OOF='all adopted v364 layers replayed; earlier controls partly proxy'))
    shutil.copyfile(baseline, OUT / 'v364_baseline.csv')
    shutil.copyfile(previous / 'submission_template.csv', OUT / 'submission_template.csv')
    manifest = dict(tasks=old['tasks'], metrics=old['metrics'], pilot_tasks=TASKS,
        rows=old['rows'], unique_exact_smiles=old['unique_exact_smiles'], baseline=best,
        official_rows_sha256=sha(raw), source_inputs=old['source_inputs'],
        sources=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in sorted(set(sources))],
        controls=records, test_labels_parsed=False, all_training_remote_only=True,
        host='connect.nmb2.seetacloud.com', port=25870)
    save(OUT / 'data_manifest.json', manifest)
    names = ('v364_graph_transformer.py', 'v370_graph_transformer.py', 'v370_run_stage5.py',
             'v370_verify_remote.py', 'v370_collect_remote.py', 'v370_bootstrap.py')
    for name in names:
        shutil.copyfile(ROOT / 'tools' / name, OUT / name)
    archive = ROOT / 'releases/v370_stage5_inputs.tar.gz'
    with tarfile.open(archive, 'w:gz') as tf:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():
                tf.add(p, arcname=p.relative_to(OUT).as_posix())
    report = dict(status='PASS', archive=archive.relative_to(ROOT).as_posix(), sha256=sha(archive),
                  bytes=archive.stat().st_size, tasks=TASKS, controls=records)
    save(ROOT / 'docs/v370_input_package.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
