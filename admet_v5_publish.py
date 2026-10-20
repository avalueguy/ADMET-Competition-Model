"""Refit confirmed candidates and make a reversible, versioned submission.

The base is the exact submitted v4 CSV predictions. The checkpoint reproduces
the new head; reproduction of the blended competition output additionally
requires the archived base predictions. No false general v4 inference claim.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from admet_v5_remote_train import fit_family,predict
from admet_features_v2 import features
from admet_submission import build
from admet_upgrade_v3 import dominates, metrics


def main():
    if os.name == 'nt' or not Path('/root/autodl-tmp/admet_20260925').is_dir():
        raise RuntimeError('Final refitting is restricted to the requested remote workspace')
    data=Path('admet_competition_data/data_benchmark'); out=Path('admet_release_v5')
    protocol=json.loads(Path('admet_v5_protocol.json').read_text()); rows=[]
    task_names=sorted(d.name for d in data.iterdir())
    for task in task_names:
        assert (Path('admet_model_outputs_v5')/task/'result.json').exists(), 'Incomplete experiments'
    out.mkdir(exist_ok=True)
    for task in task_names:
        dest=out/task;dest.mkdir(exist_ok=True)
        result=json.loads((Path('admet_model_outputs_v5')/task/'result.json').read_text())
        assert result['protocol_sha256'] == hashlib.sha256(Path('admet_v5_protocol.json').read_bytes()).hexdigest(), 'Protocol changed after experiment'
        confirm=Path('admet_v5_confirmation')/task/'confirmation.json'
        accepted=result['provisional_valid_pass'] and confirm.exists() and json.loads(confirm.read_text())['accepted']
        prior=Path('admet_release_v4')/task
        baseline=pd.read_csv(prior/f'submission_{task}.csv', dtype={'Drug_ID':str}, keep_default_na=False)
        route=dict(task=task,accepted=bool(accepted),base_submission=str(prior/f'submission_{task}.csv'),
                   base_sha256=hashlib.sha256((prior/f'submission_{task}.csv').read_bytes()).hexdigest(),
                   protocol_sha256=result['protocol_sha256'],
                   result_sha256=hashlib.sha256((Path('admet_model_outputs_v5')/task/'result.json').read_bytes()).hexdigest(),
                   confirmation_sha256=hashlib.sha256(confirm.read_bytes()).hexdigest() if confirm.exists() else None,
                   platform_score=None,baseline_platform_score=48.862655)
        if accepted:
            frames={s:pd.read_csv(data/task/(s+'.csv')) for s in ['train','valid','test']}
            n=len(frames['train'])+len(frames['valid'])
            smi=pd.concat([df.Drug for df in frames.values()]).astype(str).tolist()
            x,_,_=features(smi,Path('admet_model_outputs_v5/features'))
            with np.load(Path('admet_pretrained/embeddings')/(task+'.npz')) as z:
                assert str(z['digest']) == hashlib.sha256(json.dumps(smi).encode()).hexdigest(), 'Encoder row checksum mismatch'
                e=z['x']
            y=pd.concat([frames['train'].Y,frames['valid'].Y]).to_numpy(float)
            model=fit_family(result['chosen'],x[:n],e[:n],y,result['kind'],result['metric'],protocol)
            candidate=predict(model,x[n:],e[n:])
            bundle=dest/'candidate_model.joblib';joblib.dump(model,bundle,compress=3)
            roundtrip=predict(joblib.load(bundle),x[n:],e[n:])
            np.testing.assert_allclose(roundtrip,candidate,rtol=1e-7,atol=1e-7)
            alpha=result['alpha'];final=(1-alpha)*baseline.Y.to_numpy(float)+alpha*candidate
            assert np.isfinite(final).all()
            if result['kind']=='classification':assert np.all((final>=0)&(final<=1))
            pd.DataFrame(dict(Drug_ID=baseline.Drug_ID,Y=final)).to_csv(dest/f'submission_{task}.csv',index=False)
            pd.DataFrame(dict(Drug_ID=baseline.Drug_ID,Y=candidate)).to_csv(dest/'candidate_test_predictions.csv',index=False)
            v=pd.read_csv(Path('admet_model_outputs_v5')/task/'validation_comparison.csv')
            assert dominates(result['kind'],metrics(result['kind'],v.Y_true,v.baseline),metrics(result['kind'],v.Y_true,v.blend))
            pd.DataFrame(dict(Drug_ID=v.Drug_ID,Y_true=v.Y_true,Y_pred=v.blend)).to_csv(dest/'validation_predictions.csv',index=False)
            route.update(candidate_model='candidate_model.joblib',alpha=alpha,
                         candidate_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
                         inference_contract='frozen v4 competition predictions plus reloadable new head; not a standalone v4 arbitrary-SMILES model')
        else:
            for name in [f'submission_{task}.csv','validation_predictions.csv']:
                shutil.copy2(prior/name,dest/name)
            assert (dest/f'submission_{task}.csv').read_bytes()==(prior/f'submission_{task}.csv').read_bytes()
            route['reason']=json.loads(confirm.read_text()).get('reason','matched confirmation failed') if confirm.exists() else 'valid gate rejected'
        (dest/'selection.json').write_text(json.dumps(route,indent=2),encoding='utf8')
        rows.append(route);print(json.dumps(route),flush=True)
    pd.DataFrame(rows).to_csv(out/'routing.csv',index=False)
    artifacts=[Path('admet_v5_protocol.json'),Path('admet_legacy_recipes.json'),
               Path('admet_pretrained/chemberta_mlm/provenance.json'),Path('logs/environment.txt')]
    artifacts += sorted(Path('.').glob('admet_v5_*.py'))
    artifacts += [Path('admet_features_v2.py'),Path('admet_pretrained_features.py')]
    manifest=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in artifacts if p.exists()]
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    if any(r['accepted'] for r in rows):
        build(out,Path('admet_submissions/admet-v5-20260925.csv'))
    else:
        print('NO_PROMOTION: no task passed all gates; retain official 48.862655 submission',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
