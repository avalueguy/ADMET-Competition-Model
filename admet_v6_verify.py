"""Independent process replay and exact baseline/template alignment checks."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from admet_v6_runner import remote_only, load_task, predict_candidate, sha, save
from admet_v6_features import mixed_kernel
from admet_submission import same_id


def needs_kernel(bundle):
    return (bundle.get('family','').startswith('kernel') or
            any(needs_kernel(b) for b in bundle.get('composite',[])))


def main():
    remote_only()
    root=Path('admet_release_v6');verification=json.loads((root/'verification.json').read_text())
    if verification['status']=='NO_PROMOTION':return
    output=Path(verification['submission'])
    assert sha(output)==verification['sha256']
    submission=pd.read_csv(output,dtype={'drug_id':str},keep_default_na=False)
    template=pd.read_csv('admet_official/submission_template.csv',dtype={'drug_id':str},keep_default_na=False)
    assert submission.drop(columns='y_pred').equals(template.drop(columns='y_pred'))
    report=[]
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        task=folder.name;d=json.loads((folder/'selection.json').read_text())
        result=pd.read_csv(folder/f'submission_{task}.csv',dtype={'Drug_ID':str},keep_default_na=False)
        previous=Path('admet_release_v5')/task/f'submission_{task}.csv'
        assert sha(previous)==d['base_sha256']
        if d['accepted']:
            assert sha(folder/'model.joblib')==d['model_sha256']
            data=load_task(task);model=joblib.load(folder/'model.joblib')
            if needs_kernel(model):data['kernel']=mixed_kernel(data)
            n=len(data['frames']['train'])+len(data['frames']['valid'])
            p=predict_candidate(model,data,np.arange(n,len(data['smiles'])))
            reference=pd.read_csv(folder/'candidate_test.csv').Y.to_numpy(float)
            np.testing.assert_allclose(p,reference,rtol=3e-6,atol=3e-6)
            old=pd.read_csv(previous).Y.to_numpy(float);alpha=d['selection']['alpha']
            if d.get('domain_guard',{}).get('passed'):
                from admet_v6_domain_guard import nearest
                alpha=np.where(nearest(data['raw'],np.arange(n),np.arange(n,len(data['smiles'])))<.3,0.,alpha)
            np.testing.assert_allclose(result.Y,(1-alpha)*old+alpha*p,rtol=3e-6,atol=3e-6)
        else:assert previous.read_bytes()==(folder/f'submission_{task}.csv').read_bytes()
        piece=submission.loc[submission.task==task]
        assert all(same_id(a,b) for a,b in zip(piece.drug_id,result.Drug_ID))
        np.testing.assert_allclose(piece.y_pred,result.Y,rtol=1e-12,atol=1e-12)
        report.append(dict(task=task,accepted=d['accepted'],passed=True))
    verification.update(independent_replay=True,task_checks=report)
    save(root/'independent_verification.json',verification)
    print(json.dumps(verification),flush=True)


if __name__=='__main__':main()
