"""Reconstruct v20 evaluation predictions and read the exact scored submission."""
from pathlib import Path
import numpy as np
import pandas as pd
from admet_v20_residual import (anchor_oof as old_oof, anchor_valid as old_valid,
                               read_json, sha, metric_for, gain, save_json)
from admet_v20_rare import combine

ROOT = Path(__file__).resolve().parent
UPDATED = 'cyp2d6_substrate_carbonmangels'


def anchor_oof(task):
    y, fold, base = old_oof(task)
    if task == UPDATED:
        root = ROOT/'admet_model_outputs_v20_rare'/task
        sel = read_json(root/'decision.json')['selection']
        frame = pd.read_csv(root/'rare_oof.csv')
        np.testing.assert_array_equal(frame.Y_true,y)
        np.testing.assert_array_equal(frame.fold,fold)
        pred = frame.Y_pred.to_numpy(float)
        for f in np.unique(fold):
            mask = fold==f
            base[mask] = combine(base[mask], pred[mask], sel['weight'], sel['mode'])
    return y, fold, base


def anchor_valid(task):
    y, base = old_valid(task)
    if task == UPDATED:
        root = ROOT/'admet_model_outputs_v20_rare'/task
        sel = read_json(root/'decision.json')['selection']
        pred = pd.read_csv(root/'rare_valid.csv')
        np.testing.assert_array_equal(pred.Y_true,y)
        base = combine(base,pred.Y_pred.to_numpy(float),sel['weight'],sel['mode'])
    return y, base


def anchor_test(task):
    path = next((ROOT/'admet_submissions/v20').glob('*.csv'))
    assert sha(path)==read_json(ROOT/'admet_v21_protocol.json')['baseline_sha256']
    frame = pd.read_csv(path,dtype=str,keep_default_na=False)
    subset = frame.loc[frame.task.eq(task)]
    np.testing.assert_array_equal(subset.sample_idx.astype(int),np.arange(len(subset)))
    return subset.y_pred.to_numpy(float)
