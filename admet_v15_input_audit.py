"""Read-only provenance and split audit of reused OOF inputs and repeat anchors."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from admet_v15_runner import names,read,anchor,sha,OUT
from admet_v15_models import load_task
from admet_v6_runner import save,remote_only,partition


def main():
    remote_only();result=[]
    for task in names():
        s=read(OUT/task/'shortlist.json');assert s['protocol_sha256']==sha('admet_v15_protocol.json')
        b=anchor(task);d=load_task(task);n=len(d['frames']['train'])
        np.testing.assert_array_equal(b.row,np.arange(n));np.testing.assert_array_equal(b.fold,partition(task,d))
        np.testing.assert_allclose(b.Y_true,d['frames']['train'].Y)
        for source in s['provenance']:
            assert sha(source['path'])==source['sha256']
            version=source['name'].split(':')[0]
            assert sha('admet_'+version+'_protocol.json')==source['recipe_sha256']
        for fold in range(3):
            i=np.flatnonzero(b.fold!=fold);j=np.flatnonzero(b.fold==fold)
            assert not set(d['canonical'][i])&set(d['canonical'][j])
            assert not set(d['groups'][i])&set(d['groups'][j])
        route=read(Path('admet_release_v14')/task/'selection.json')
        if route['version']!=11:
            i,j=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=692509).split(np.arange(n),groups=d['groups'][:n]))
            old=pd.read_csv(Path('admet_model_outputs_v'+str(route['version']))/task/'repeat_comparison.csv')
            np.testing.assert_array_equal(old.row,j);np.testing.assert_allclose(old.Y_true,d['frames']['train'].Y.to_numpy()[j])
        result.append(dict(task=task,provenance_files=len(s['provenance']),oof_molecule_overlap=0,oof_scaffold_overlap=0,passed=True))
    report=dict(status='PASSED',tasks=result,scope='Reused source hashes, matched target rows/labels/folds and cached v12/v13 repeat alignment; not an unbiased validation claim')
    save('admet_diagnostics/v15_input_audit.json',report)
    print(json.dumps(dict(status='PASSED',tasks=len(result),provenance_files=sum(r['provenance_files'] for r in result))))


if __name__=='__main__':main()
