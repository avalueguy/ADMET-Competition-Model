"""Bound floating point reconstruction noise in v233; retain exact model predictions."""
import joblib
import numpy as np
from scipy.stats import rankdata

import v351_verify_combination_models as original

experiment = original.experiment
ROOT = original.ROOT


def compare(a, b, changes, name="root"):
    if isinstance(a, dict):
        assert set(a) == set(b), name
        for key in a:
            compare(a[key], b[key], changes, name + "." + key)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), name
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, changes, name + f"[{i}]")
    else:
        x, y = np.asarray(a), np.asarray(b)
        assert x.shape == y.shape, name
        if x.dtype.kind in "fc" and y.dtype.kind in "fc":
            np.testing.assert_allclose(x, y, rtol=2e-13, atol=2e-14, equal_nan=True, err_msg=name)
            finite = np.isfinite(x) & np.isfinite(y)
            error = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.
            if error:
                changes.append(dict(field=name, max_absolute_error=error))
        else:
            np.testing.assert_array_equal(x, y, err_msg=name)


def check(d, k, part, record, mods):
    if record["version"] != 233:
        return original.check(d, k, part, record, mods)
    model, config = mods[233], record["config"]
    expected, provenance = experiment.cached(d, k, part, record, mods)
    folder = model.OUT / "models" / d["tasks"][k] / part / record["key"]
    a = model.design(d, k, part, config)
    differences = []
    compare(a["state"], joblib.load(folder / "transform.joblib"), differences, "saved_transform")
    net = joblib.load(folder / "model.joblib")
    replay = model.infer(net, a["qx"], d["metrics"][k], a["state"])
    np.testing.assert_allclose(replay, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(rankdata(replay), rankdata(expected))
    fit, query = experiment.v.split(d, k, part)
    labels = d["y"].copy()
    try:
        d["y"][(d["tid"] == k) & ~np.isin(np.arange(len(labels)), fit)] = 999999.
        d.pop("_v96_assays", None)
        altered = model.design(d, k, part, config)
        compare(a, altered, differences, "nonfit_label_isolation")
        altered_p = model.infer(net, altered["qx"], d["metrics"][k], altered["state"])
        np.testing.assert_allclose(altered_p, replay, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(rankdata(altered_p), rankdata(replay))
    finally:
        d["y"][:] = labels
        d.pop("_v96_assays", None)
    hashes = {p.relative_to(ROOT).as_posix():experiment.sha(p) for p in folder.iterdir() if p.is_file()}
    return dict(task=d["tasks"][k], part=part, version=233, rows=len(query),
        saved_prediction=provenance, saved_model_reload=True, nonfit_target_label_isolation=True,
        exact_split_alignment=True, model_files=hashes,
        replay_adapter="tools/v358_kernel_replay_adapter.py",
        transform_float_rtol=2e-13, transform_float_atol=2e-14,
        all_integer_string_and_split_fields_exact=True, prediction_atol=1e-12,
        prediction_max_abs_error=float(np.max(np.abs(replay - expected))),
        all_prediction_ranks_exact=True, numerical_differences=differences)
