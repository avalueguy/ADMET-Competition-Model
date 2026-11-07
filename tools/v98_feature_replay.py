"""Rebuild selected label-free conformers from independent official row traversal."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v98_conformer_features as f

def main():
    d=f.v.load();z=np.load(f.OUT/'conformers.npz');audit=json.loads((f.OUT/'audit.json').read_text())
    assert audit['signature']==f.sig()==str(z['signature']) and f.v.b.sha(f.OUT/'conformers.npz')==audit['sha256']
    all_smiles=[None]*len(z['x']);cursor=0;pool={p:[] for p in ('train','valid','test')}
    for task in d['tasks']:
        for part in pool:
            rows=([{'Drug':r['smiles']} for r in d['template'] if r['task']==task] if part=='test'
                else f.v.b.table(ROOT/'admet_competition_data/data_benchmark'/task/(part+'.csv')))
            for row in rows:
                index=int(d['mi'][cursor]);assert all_smiles[index] in (None,row['Drug']);all_smiles[index]=row['Drug']
                pool[part].append(index);cursor+=1
    assert hashlib.sha256(json.dumps(all_smiles).encode()).hexdigest()==str(z['smiles_sha256'])
    rng=np.random.default_rng(989926);ids=sorted(set(int(i) for p in pool for i in rng.choice(np.unique(pool[p]),4,replace=False)))
    records=[]
    for i in ids:
        _,vector,status=f.build_one((i,all_smiles[i]));vector[-1]=float(status=='success')
        assert status==str(z['status'][i]);np.testing.assert_allclose(vector,z['x'][i],atol=1e-5,rtol=1e-6,equal_nan=True)
        records.append(dict(index=i,status=status,max_abs_difference=float(np.nanmax(np.abs(vector-z['x'][i])))))
    report=dict(status='PASS',signature=f.sig(),feature_sha256=audit['sha256'],official_smiles_sha256=str(z['smiles_sha256']),
        rows_reconstructed=cursor,selection='4 fixed random molecule IDs per official split, independent of labels/outcomes',records=records,test_labels_used=False)
    f.v.b.save(f.OUT/'replay.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
