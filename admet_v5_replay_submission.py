"""Replay a release from immutable base predictions and saved heads, without fitting."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from admet_features_v2 import features
from admet_pretrained_features import Encoder
from admet_submission import build
from admet_v5_remote_train import predict


def main():
    if os.name=='nt' or not Path('/root/autodl-tmp/admet_20260925').is_dir():
        raise RuntimeError('GPU replay is restricted to the requested remote workspace')
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    source=Path('admet_release_v5')
    report=json.loads((source/'verification.json').read_text())
    assert report['status']=='VERIFIED'
    assert report['accepted'], 'No new candidate was promoted'
    assert not args.output.exists(), 'Choose a fresh replay output path'
    work=args.output.parent/(args.output.stem+'_parts')
    assert not work.exists(), 'Replay scratch directory already exists'
    work.mkdir(parents=True)
    encoder=None
    replay_audit=[]
    for cfg in sorted(source.glob('*/selection.json')):
        r=json.loads(cfg.read_text());task=r['task']
        base=Path(r['base_submission'])
        assert hashlib.sha256(base.read_bytes()).hexdigest()==r['base_sha256']
        dest=work/task;dest.mkdir()
        target=dest/f'submission_{task}.csv'
        if not r['accepted']:
            shutil.copy2(base,target)
            continue
        model_path=source/task/r['candidate_model']
        assert hashlib.sha256(model_path.read_bytes()).hexdigest()==r['candidate_sha256']
        frames={s:pd.read_csv(Path('admet_competition_data/data_benchmark')/task/(s+'.csv')) for s in ['train','valid','test']}
        all_smiles=pd.concat([df.Drug for df in frames.values()]).astype(str).tolist()
        n=len(frames['train'])+len(frames['valid'])
        xx,_,_=features(all_smiles)
        x=xx[n:]
        if encoder is None:
            encoder=Encoder()
        # Preserve original ordering and padding batches. Tree thresholds can
        # amplify sub-micro floating point embedding differences across batches.
        ee,_=encoder.encode(all_smiles)
        e=ee[n:]
        p=predict(joblib.load(model_path),x,e)
        old=pd.read_csv(base,dtype={'Drug_ID':str},keep_default_na=False)
        old.Y=(1-r['alpha'])*old.Y.to_numpy(float)+r['alpha']*p
        expected=pd.read_csv(source/task/f'submission_{task}.csv').Y.to_numpy(float)
        np.testing.assert_allclose(old.Y,expected,atol=1e-5,rtol=1e-5)
        replay_audit.append(dict(task=task,max_prediction_difference=float(np.max(np.abs(old.Y.to_numpy(float)-expected))),
                                 context='original full task ordering and batch padding'))
        old.to_csv(target,index=False)
    build(work,args.output)
    (source/'smiles_replay_verification.json').write_text(json.dumps(dict(status='VERIFIED',tasks=replay_audit),indent=2),encoding='utf8')
    print(args.output)


if __name__=='__main__':
    torch.set_num_threads(6)
    with threadpool_limits(limits=6):main()
