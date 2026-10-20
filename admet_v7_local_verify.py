"""Lightweight local evidence retrieval and exact upload format validation; no training."""
import json
import hashlib
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    workspace=Path.cwd().resolve();allowed={'admet_model_outputs_v7','admet_release_v7','admet_submissions','admet_diagnostics','logs'}
    with tarfile.open('admet_v7_review.tar.gz','r:gz') as tf:
        for m in tf.getmembers():
            p=Path(m.name)
            if not p.parts or p.parts[0] not in allowed or not m.isfile():raise ValueError(m.name)
            if p.parts[0]=='admet_submissions' and (len(p.parts)<2 or p.parts[1]!='v7'):raise ValueError(m.name)
            if p.parts[0]=='admet_diagnostics' and not p.name.startswith('v7_'):raise ValueError(m.name)
            target=(workspace/p).resolve()
            if not target.is_relative_to(workspace):raise ValueError(m.name)
            if p.parts[0]=='logs':target=workspace/'admet_diagnostics/v7_remote_logs'/p.name
            target.parent.mkdir(parents=True,exist_ok=True);content=tf.extractfile(m).read()
            if p.parts[0]=='admet_submissions' and target.exists() and target.read_bytes()!=content:raise ValueError('different existing submission')
            target.write_bytes(content)
    root=Path('admet_release_v7');v=json.loads((root/'independent_verification.json').read_text(encoding='utf8'))
    if v['status']=='NO_PROMOTION':print(json.dumps(v));return
    assert v['status']=='VERIFIED' and v['independent_replay']
    file=Path(v['submission']);blob=file.read_bytes()
    assert hashlib.sha256(blob).hexdigest()==v['sha256']
    assert blob.startswith(b'\xef\xbb\xbf') and blob.count(b'\r\n')==16380
    assert blob.count(b'task,sample_idx,drug_id,smiles,y_pred,metric')==1
    new=pd.read_csv(file,dtype=str,keep_default_na=False);template=pd.read_csv('admet_official/submission_template.csv',dtype=str,keep_default_na=False)
    assert new.drop(columns='y_pred').equals(template.drop(columns='y_pred'))
    assert list(new.columns)==['task','sample_idx','drug_id','smiles','y_pred','metric']
    pred=pd.to_numeric(new.y_pred).to_numpy(float);assert np.isfinite(pred).all()
    classification=new.metric.isin(['AUROC','AUPRC']).to_numpy();assert ((pred[classification]>=0)&(pred[classification]<=1)).all()
    oldfile=next(Path('admet_submissions/v6').glob('*.csv'));assert hashlib.sha256(oldfile.read_bytes()).hexdigest()=='8a677b09bbc8f263ef9ae27055cd2cc88de56a0ed7c7991de67a5256cdb628a4'
    old=pd.read_csv(oldfile,dtype=str,keep_default_na=False);assert old.drop(columns='y_pred').equals(new.drop(columns='y_pred'))
    changed=[];unchanged=[]
    for task in new.task.unique():
        mask=new.task.eq(task);is_changed=not np.array_equal(pred[mask],pd.to_numeric(old.loc[mask,'y_pred']).to_numpy(float))
        (changed if is_changed else unchanged).append(task)
    assert set(changed)==set(v['accepted'])
    result=dict(status='VERIFIED_LOCAL_AND_REMOTE',submission=str(file),sha256=v['sha256'],rows=len(new),columns=len(new.columns),tasks=new.task.nunique(),
        changed=changed,unchanged=unchanged,encoding='UTF-8 BOM',line_endings='CRLF',header_count=1,platform_score=None)
    (root/'local_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
