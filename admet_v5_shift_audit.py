"""Compare validation/test structure shift without reading any test targets."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from admet_features_v2 import features


def nearest(train,query):
    a=torch.as_tensor(train[:,:2048],device='cuda',dtype=torch.float32)
    ac=a.sum(1)
    out=[]
    for start in range(0,len(query),256):
        b=torch.as_tensor(query[start:start+256,:2048],device='cuda',dtype=torch.float32)
        dot=b@a.T
        sim=dot/(b.sum(1)[:,None]+ac[None,:]-dot).clamp(min=1)
        out.extend(sim.max(1).values.cpu().numpy())
    return np.asarray(out)


def main():
    if os.name=='nt' or not Path('/root/autodl-tmp/admet_20260925').is_dir():
        raise RuntimeError('Remote CUDA audit required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path('admet_competition_data/data_benchmark')
    out=Path('admet_diagnostics');out.mkdir(exist_ok=True)
    rows=[]
    with torch.inference_mode():
        for task in sorted(p.name for p in root.iterdir() if p.is_dir()):
            f={s:pd.read_csv(root/task/(s+'.csv')) for s in ['train','valid','test']}
            n,nv=len(f['train']),len(f['valid'])
            smi=pd.concat([df.Drug for df in f.values()]).astype(str).tolist()
            # Reuse any existing per-worker deterministic feature cache.
            x=None
            for cache in ['admet_model_outputs_v5/features','admet_model_outputs_v5_tail/features','admet_model_outputs_v5_middle/features']:
                for p in Path(cache).glob('*.npz'):
                    with np.load(p) as z:
                        if len(z['x'])==len(smi):
                            # Only trust a content-keyed feature cache via its API.
                            from admet_features_v2 import SCHEMA
                            key=hashlib.sha256(json.dumps([SCHEMA,smi],sort_keys=True).encode()).hexdigest()
                            if p.stem==key:
                                x,canon,scaffold=features(smi,Path(cache));break
                if x is not None:break
            if x is None:x,canon,scaffold=features(smi,Path('admet_model_outputs_v5/features'))
            for split,start,end in [('valid',n,n+nv),('test',n+nv,len(smi))]:
                sims=nearest(x[:n],x[start:end])
                p=Path('admet_release_v4')/task/('validation_predictions.csv' if split=='valid' else f'submission_{task}.csv')
                pred=pd.read_csv(p)
                yp=pred['Y_pred' if split=='valid' else 'Y'].to_numpy(float)
                row=dict(task=task,split=split,rows=end-start,mean_nearest=float(sims.mean()),
                         fraction_similarity_below_03=float(np.mean(sims<.3)),
                         nearest_q10=float(np.quantile(sims,.1)),nearest_median=float(np.median(sims)),
                         nearest_q90=float(np.quantile(sims,.9)),prediction_mean=float(yp.mean()),
                         prediction_std=float(yp.std()),
                         invalid_smiles=sum(s.startswith('INVALID:') for s in canon[start:end]))
                if split=='valid':
                    y=f['valid'].Y.to_numpy(float)
                    row['observed_label_mean']=float(y.mean())
                    row['observed_label_std']=float(y.std())
                rows.append(row)
            print(json.dumps(dict(task=task,valid_nearest=rows[-2]['mean_nearest'],test_nearest=rows[-1]['mean_nearest'])),flush=True)
    pd.DataFrame(rows).to_csv(out/'v5_structure_shift.csv',index=False)
    (out/'v5_structure_shift_notes.txt').write_text(
        'Nearest Tanimoto uses radius-2 2048-bit Morgan fingerprints, identical task train reference for valid and test. '
        'No target labels from test were read. Structural distance alone cannot establish task performance or causes of platform score.\n',encoding='utf8')


if __name__=='__main__':main()
