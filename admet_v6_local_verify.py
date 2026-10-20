"""CPU-only archive extraction and final file validation on the user machine."""
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    workspace=Path.cwd().resolve()
    # Extract only result directories, never remote script copies or model weights.
    allowed={'admet_model_outputs_v6','admet_release_v6','admet_submissions','admet_diagnostics','logs'}
    with tarfile.open('admet_v6_review.tar.gz','r:gz') as tf:
        for member in tf.getmembers():
            path=Path(member.name)
            if not path.parts or path.parts[0] not in allowed:continue
            target=(workspace/path).resolve()
            if not target.is_relative_to(workspace) or not member.isfile():raise ValueError(member.name)
            if path.parts[0]=='logs':target=workspace/'admet_diagnostics'/'v6_remote_logs'/path.name
            if target.exists() and target.read_bytes()!=tf.extractfile(member).read():
                if path.parts[0] not in {'admet_model_outputs_v6','admet_release_v6','logs','admet_diagnostics'}:
                    raise ValueError('Refusing to overwrite different submission: '+member.name)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(tf.extractfile(member).read())
    verify=json.loads(Path('admet_release_v6/independent_verification.json').read_text(encoding='utf8'))
    file=Path(verify['submission']);blob=file.read_bytes()
    assert hashlib.sha256(blob).hexdigest()==verify['sha256']
    assert blob.startswith(b'\xef\xbb\xbf')
    assert blob.count(b'task,sample_idx,drug_id,smiles,y_pred,metric')==1
    assert blob.count(b'\r\n')==16380
    new=pd.read_csv(file,dtype=str,keep_default_na=False)
    template=pd.read_csv('admet_official/submission_template.csv',dtype=str,keep_default_na=False)
    assert new.drop(columns='y_pred').equals(template.drop(columns='y_pred'))
    p=pd.to_numeric(new.y_pred).to_numpy(float)
    assert np.isfinite(p).all()
    is_class=new.metric.isin(['AUROC','AUPRC']).to_numpy()
    assert ((p[is_class]>=0)&(p[is_class]<=1)).all()
    old=pd.read_csv('admet_submissions/我从山北来带着兰花草-小分子药物预测技术创新赛-20260925-平台上传版.csv',dtype=str,keep_default_na=False)
    assert new.drop(columns='y_pred').equals(old.drop(columns='y_pred'))
    changed=[];unchanged=[]
    for task in new.task.unique():
        mask=new.task.eq(task)
        diff=np.max(np.abs(p[mask]-pd.to_numeric(old.loc[mask,'y_pred']).to_numpy(float)))
        (changed if diff else unchanged).append(task)
    assert set(changed)==set(verify['accepted'])
    result=dict(status='VERIFIED_LOCAL_AND_REMOTE',sha256=verify['sha256'],rows=len(new),columns=len(new.columns),
                tasks=new.task.nunique(),changed=changed,unchanged=unchanged,header_count=1,
                encoding='UTF-8 BOM',line_endings='CRLF',remote_independent_replay=True)
    Path('admet_release_v6/local_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    state=json.loads(Path('admet_v6_remote_run.json').read_text())
    state.update(status='complete_submission_verified_not_uploaded',submission=str(file),sha256=verify['sha256'],
                 changed_task_count=len(changed),unchanged_task_count=len(unchanged))
    Path('admet_v6_remote_run.json').write_text(json.dumps(state,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
