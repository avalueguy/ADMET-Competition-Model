"""Independent receipts for the completed auxiliary residual experiment."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('experiment', ROOT/'v390_auxiliary_residual.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    protocol = json.loads((ROOT/'protocol.json').read_text())
    assert protocol['source_sha256'] == sha(m.__file__)
    assert json.loads((ROOT/'status.json').read_text())['stage'] == 'COMPLETE'
    d = m.load()
    teachers = []
    for path in sorted((ROOT/'teachers').rglob('*.joblib')):
        task, part = path.parts[-3:-1]
        s = joblib.load(path)
        source, query = s['source'], s['query']
        assert np.all(d['tid'][source] != d['tasks'].index(task))
        assert np.all(d['split'][source] < (2 if part == 'final' else 1))
        assert not np.intersect1d(d['rg'][source], d['rg'][query]).size
        # Rebuild the full teacher feature transform independently from its source.
        x, qx, columns, scaling = m.teacher_features(d, source, query)
        np.testing.assert_array_equal(columns, s['columns'])
        for k in ('median', 'scale'):
            np.testing.assert_array_equal(scaling[k], s['scaling'][k])
        rebuilt = []
        for state in s['models']:
            if state['empty']:
                rebuilt.append(np.zeros(len(query)))
            else:
                assert np.all(d['tid'][state['source']] == state['task'])
                # NumPy multiplication differs slightly from the CUDA training product.
                rebuilt.append(qx @ state['coefficient'])
        panel = np.column_stack(rebuilt)
        error = float(abs(panel-s['panel']).max())
        np.testing.assert_allclose(panel, s['panel'], rtol=2e-4, atol=2e-4)
        teachers.append(dict(path=path.relative_to(ROOT).as_posix(),
                             max_replay_error=error, source_scaffold_isolation=True,
                             target_task_excluded=True, sha256=sha(path)))
    curves = []
    for path in sorted((ROOT/'curves').rglob('state.joblib')):
        task, part, config = path.parts[-4:-1]
        a = m.design(d, task, part)
        for split in a['splits']:
            assert not np.intersect1d(d['rg'][split['train']], d['rg'][split['held']]).size
        state = joblib.load(path)
        for k in ('fit', 'query'):
            np.testing.assert_array_equal(a[k], state[k])
        assert len(state['models']) == 2
        assert all(net.n_estimators == 3072 for net in state['models'])
        b = next(v for v in m.BASES if v['name'] == config)
        errors = []
        for horizon in m.HORIZONS:
            prediction = m.replay(d, task, part, b, horizon, a)
            np.testing.assert_array_equal(prediction, state['predictions'][horizon])
            changed = dict(d, y=d['y'].copy())
            changed['y'][a['query']] = -987654.
            # Query labels are not consumed by feature/replay construction.
            achanged = m.design(changed, task, part)
            rp = m.replay(changed, task, part, b, horizon, achanged)
            np.testing.assert_array_equal(rp, prediction)
            errors.append(float(abs(prediction-state['predictions'][horizon]).max()))
        curves.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                           replay_max_error=max(errors), query_label_perturbation_pass=True))
    development = json.loads((ROOT/'development.json').read_text())['records']
    frozen = json.loads((ROOT/'frozen_selection.json').read_text())['selected']
    for row, chosen in zip(development, frozen):
        best = max(row['choices'], key=lambda v: (v['objective'], -v['weight'], -v['horizon'], v['base']))
        assert chosen == best
    value = dict(status='PASS', teachers=teachers, curves=curves,
        all_curve_count=len(curves), actual_seed_fit_count=2*len(curves),
        expected_development_seed_fits=60, complete_development_seed_fits=
            sum(2 for r in curves if '/fold0/' in r['path'] or '/fold1/' in r['path']),
        selection_recomputed=True, source_sha256=sha(m.__file__),
        protocol_sha256=sha(ROOT/'protocol.json'), verified_at_unix=time.time())
    assert value['complete_development_seed_fits'] == 60
    m.save(ROOT/'verification.json', value)
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and
                   p.name not in ('result_manifest.json', 'results.tar.gz', 'training.log') and
                   '__pycache__' not in p.parts)
    m.save(ROOT/'result_manifest.json', dict(files=[dict(path=p.relative_to(ROOT).as_posix(),
        sha256=sha(p), bytes=p.stat().st_size) for p in files]))
    print(json.dumps(dict(status='PASS', curves=len(curves), teachers=len(teachers))))


if __name__ == '__main__':
    main()
