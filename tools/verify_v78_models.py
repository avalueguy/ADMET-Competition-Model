"""Independently replay every v78 model with saved fit rows and transformations."""
from pathlib import Path
import json
import sys
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import admet_v77_ranking as data
import admet_v78_median as v


def main():
    d = data.load_data()
    d['folds'] = np.load(ROOT / 'recovered_history/legacy_v24_folds.npy')
    assert data.sha(ROOT / 'recovered_history/legacy_v24_folds.npy') == '3a24d8a1359df47c4197bff62a80809bbdc42f53db22c5dd1f478db2a95a4b98'
    with np.load(ROOT / 'admet_model_outputs_v73/features/views.npz') as z:
        v.PRIOR['values'] = z['priors'].copy()
    records = []
    for file in sorted(v.OUT.rglob('model.ubj')):
        folder = file.parent
        meta = json.loads((folder / 'metadata.json').read_text())
        assert meta['signature'] == v.signature()
        assert meta['model_sha256'] == data.sha(file)
        assert meta['predictions_sha256'] == data.sha(folder / 'predictions.npz')
        task, part, view, method, seed_dir = folder.relative_to(v.OUT).parts
        assert meta['task'] == task and meta['view'] == view and meta['method'] == method
        k = d['tasks'].index(task)
        rows, q = data.rows(d, k, part)
        if seed_dir == 'inner':
            rows, q = data.inner(d, rows, k)
        else:
            assert int(seed_dir) == meta['seed'] and 'early_stopping_rounds' not in meta['config']
        cache = np.load(folder / 'predictions.npz')
        assert str(cache['signature']) == v.signature()
        np.testing.assert_array_equal(cache['fit'], rows)
        np.testing.assert_array_equal(cache['query'], q)
        assert np.all(d['split'][rows] < 2) and not set(rows) & set(q)
        if part != 'final':
            assert not set(d['groups'][rows]) & set(d['groups'][q])
            assert not set(d['canonical'][rows]) & set(d['canonical'][q])
        raw = d['y'][rows] - v.prior(d, task, rows)
        center = float(np.median(raw))
        scale = max(float(np.subtract(*np.quantile(raw, [.75, .25]))), .01)
        assert center == meta['target']['center'] and scale == meta['target']['scale']
        model = xgb.XGBRegressor()
        model.load_model(file)
        actual = meta['actual_trees']
        assert model.get_booster().num_boosted_rounds() == meta['stopped_trees'] >= actual
        if seed_dir != 'inner':
            assert model.get_booster().num_boosted_rounds() == actual
            if part != 'final':
                assert actual == json.loads((folder.parent / 'inner/metadata.json').read_text())['actual_trees']
        def predict(qq):
            x = d['views'][view][d['mi'][qq]]
            p = model.predict(x, iteration_range=(0, actual)).astype(float) * scale + center + v.prior(d, task, qq)
            return p.clip(0, 100) if task == 'ppbr_az' else p
        p = predict(q)
        np.testing.assert_allclose(p, cache['p'], atol=1e-7, rtol=1e-7)
        sample = np.unique([0, len(q) // 2, len(q) - 1])
        np.testing.assert_allclose(p[sample], predict(q[sample]), atol=1e-7, rtol=1e-7)
        records.append(dict(path=file.relative_to(ROOT).as_posix(), status='PASS',
            model_sha256=data.sha(file), query_rows=len(q), fit_rows=len(rows), trees=actual,
            max_error=float(np.max(np.abs(p - cache['p'])))))
    ensembles = 0
    for path in v.OUT.rglob('ensemble.npz'):
        with np.load(path) as z:
            parts = [np.load(path.parent / str(seed) / 'predictions.npz') for seed in v.SEEDS]
            for p in parts:
                np.testing.assert_array_equal(p['query'], z['query'])
            expected = np.mean([p['p'] for p in parts], axis=0)
            np.testing.assert_array_equal(z['p'], expected)
        ensembles += 1
    result = dict(status='PASS', signature=v.signature(), models=len(records), ensembles=ensembles,
        model_records=records, test_labels_used=False, test_csv_opened=False,
        transformations_recomputed_from_fit_only=True, query_batch_invariance=True)
    data.save(v.OUT / 'model_verification.json', result)
    print(json.dumps({k: value for k, value in result.items() if k != 'model_records'}), flush=True)


if __name__ == '__main__':
    main()
