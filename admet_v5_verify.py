"""Replay saved candidates and audit a versioned submission on the remote host."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from admet_features_v2 import features, SCHEMA
from admet_submission import same_id
from admet_upgrade_v3 import dominates, metrics
from admet_v5_remote_train import Preprocessor, predict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if os.name == 'nt' or not Path('/root/autodl-tmp/admet_20260925').is_dir():
        raise RuntimeError('Run replay only on the requested remote server')
    # The first training process was launched as a script. Remap only its known
    # local Preprocessor class, then save using the stable importable module.
    setattr(sys.modules['__main__'], 'Preprocessor', Preprocessor)
    root = Path('admet_competition_data/data_benchmark')
    output = Path('admet_release_v5')
    tasks = sorted(d.name for d in root.iterdir() if d.is_dir())
    template = pd.read_csv('admet_official/submission_template.csv', keep_default_na=False,
                           dtype={'drug_id': str})
    submission = Path('admet_submissions/admet-v5-20260925.csv')
    rows = []
    expected = template.copy()
    expected['y_pred'] = np.nan
    for task in tasks:
        run = Path('admet_model_outputs_v5') / task
        route = json.loads((output / task / 'selection.json').read_text())
        result = json.loads((run / 'result.json').read_text())
        assert sha(run/'result.json') == route['result_sha256']
        if route['confirmation_sha256'] is not None:
            assert sha(Path('admet_v5_confirmation')/task/'confirmation.json') == route['confirmation_sha256']
        frames = {s: pd.read_csv(root/task/(s+'.csv')) for s in ['train','valid','test']}
        n, nv = len(frames['train']), len(frames['valid'])
        smiles = pd.concat([df.Drug for df in frames.values()]).astype(str).tolist()
        x, _, _ = features(smiles, Path('admet_model_outputs_v5/features'))
        # Workers use separate caches. AvgIpc is numerically unstable for a few
        # large molecules, so validation replay must use the actual training
        # feature artifact rather than silently regenerating it elsewhere.
        feature_key = hashlib.sha256(json.dumps([SCHEMA,smiles],sort_keys=True).encode()).hexdigest()
        origin = Path('admet_model_outputs_v5/features')/(feature_key+'.npz')
        worker = ('middle' if task in {'lipophilicity_astrazeneca','ld50_zhu','hia_hou','herg','half_life_obach'}
                  else 'tail' if task in {'vdss_lombardo','solubility_aqsoldb','ppbr_az','pgp_broccatelli'} else None)
        if worker is not None:
            candidate_origin = Path('admet_model_outputs_v5_'+worker+'/features')/(feature_key+'.npz')
            if candidate_origin.exists():
                origin = candidate_origin
        with np.load(origin) as z:
            validation_x = z['x']
        mismatch = ~np.isclose(x,validation_x,atol=0,rtol=0,equal_nan=True)
        with np.load(Path('admet_pretrained/embeddings')/(task+'.npz')) as z:
            assert str(z['digest']) == hashlib.sha256(json.dumps(smiles).encode()).hexdigest()
            e = z['x']
        checkpoint = run/'validation_model.joblib'
        model = joblib.load(checkpoint)
        pv = predict(model, validation_x[n:n+nv], e[n:n+nv])
        validation = pd.read_csv(run/'validation_comparison.csv')
        np.testing.assert_allclose(pv, validation.candidate, atol=1e-7, rtol=1e-7)
        # Repair the serialization name without fitting or altering predictions.
        joblib.dump(model, checkpoint, compress=3)
        row = dict(task=task, accepted=route['accepted'], valid_model_replayed=True,
                   validation_model_sha256=sha(checkpoint),
                   training_features=str(origin),training_features_sha256=sha(origin),
                   regenerated_feature_differences=int(mismatch.sum()))
        final_path = output/task/f'submission_{task}.csv'
        base_path = Path('admet_release_v4')/task/f'submission_{task}.csv'
        assert sha(base_path) == route['base_sha256']
        final = pd.read_csv(final_path, dtype={'Drug_ID':str}, keep_default_na=False)
        base = pd.read_csv(base_path, dtype={'Drug_ID':str}, keep_default_na=False)
        yy = final.Y.to_numpy(float)
        if route['accepted']:
            confirmation = json.loads((Path('admet_v5_confirmation')/task/'confirmation.json').read_text())
            assert result['provisional_valid_pass'] and confirmation['accepted']
            assert all(f['passed'] for f in confirmation['folds'])
            assert dominates(result['kind'], result['valid_before'], result['valid_blend'])
            model_path = output/task/'candidate_model.joblib'
            assert sha(model_path) == route['candidate_sha256']
            pp = predict(joblib.load(model_path), x[n+nv:], e[n+nv:])
            stored = pd.read_csv(output/task/'candidate_test_predictions.csv')
            np.testing.assert_allclose(pp, stored.Y, rtol=1e-7, atol=1e-7)
            blended = (1-route['alpha'])*base.Y.to_numpy(float)+route['alpha']*pp
            np.testing.assert_allclose(yy, blended, rtol=1e-7, atol=1e-7)
            row['refit_model_replayed'] = True
        else:
            assert final_path.read_bytes() == base_path.read_bytes()
            row['baseline_preserved_byte_for_byte'] = True
        part = template.loc[template.task == task]
        assert len(part) == len(final) == len(frames['test'])
        assert all(same_id(a,b) for a,b in zip(part.drug_id, final.Drug_ID))
        assert np.isfinite(yy).all()
        if result['kind'] == 'classification':
            assert np.all((yy >= 0) & (yy <= 1))
        expected.loc[part.index,'y_pred'] = yy
        rows.append(row)
        print(json.dumps(row), flush=True)
    accepted = [r['task'] for r in rows if r['accepted']]
    if accepted:
        actual = pd.read_csv(submission, dtype={'drug_id':str}, keep_default_na=False)
        assert len(actual) == 16379 and list(actual.columns) == list(template.columns)
        assert actual.drop(columns='y_pred').equals(template.drop(columns='y_pred'))
        np.testing.assert_allclose(actual.y_pred, expected.y_pred, rtol=1e-14, atol=1e-14)
        submission_hash = sha(submission)
    else:
        assert not submission.exists(), 'A stale submission must not be offered as a new result'
        submission_hash = None
    report = dict(status='VERIFIED', accepted=accepted, tasks=rows, rows=16379,
                  submission_sha256=submission_hash, platform_score=None,
                  limitation='Scaffold CV and reused valid are not blind platform evaluation')
    (output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(dict(status='VERIFIED',accepted=accepted,submission_sha256=submission_hash)),flush=True)


if __name__ == '__main__':
    torch.set_num_threads(6)
    with threadpool_limits(limits=6):
        main()
