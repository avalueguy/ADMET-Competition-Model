"""Read-only coverage audit for official cross-assay chemical identities."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from rdkit import Chem,RDLogger,rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize as std
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v95_graph_specialists as v


def identities(smiles):
    RDLogger.DisableLog('rdApp.*')
    try:
        mol=v.old_graph.chemistry.parse_molecule(smiles)[0]
        parent=std.LargestFragmentChooser().choose(mol)
        neutral=std.Uncharger().uncharge(parent)
        taut=std.TautomerEnumerator().Canonicalize(neutral)
        return tuple(Chem.MolToSmiles(m,isomericSmiles=False) for m in (parent,neutral,taut))
    except Exception:
        mol=v.old_graph.chemistry.parse_molecule(smiles)[0]
        key=Chem.MolToSmiles(mol,isomericSmiles=False)
        return (key,key,key)


def main():
    d=v.load();path=v.OUT/'identity_audit';path.mkdir(parents=True,exist_ok=True)
    smiles=[None]*len(d['views']['physchem']);cursor=0
    for task in d['tasks']:
        for part in ('train','valid','test'):
            rows=([{'Drug':r['smiles']} for r in d['template'] if r['task']==task] if part=='test'
                  else v.b.table(ROOT/'admet_competition_data/data_benchmark'/task/(part+'.csv')))
            for row in rows:
                i=d['mi'][cursor];assert smiles[i] in (None,row['Drug']);smiles[i]=row['Drug'];cursor+=1
    expected=json.loads((ROOT/'admet_model_outputs_v67/features/official_chemistry.json').read_text())['smiles_sha256']
    assert v.hashlib.sha256(json.dumps(smiles).encode()).hexdigest()==expected
    file=path/'identities.npz'
    if file.exists():
        z=np.load(file);keys=z['keys']
    else:
        started=time.monotonic()
        with ProcessPoolExecutor(max_workers=8) as pool:keys=np.array(list(pool.map(identities,smiles,chunksize=64)))
        np.savez_compressed(file,keys=keys,smiles_sha256=expected,rdkit=rdBase.rdkitVersion)
        v.emit(stage='identities',seconds=time.monotonic()-started)
    records=[]
    for k,task in enumerate(d['tasks']):
        source=np.flatnonzero((d['tid']!=k)&(d['split']==0))
        q=np.flatnonzero((d['tid']==k)&(d['split']==1))
        old=set(d['canonical'][source]);mask=np.isin(d['canonical'][q],list(old));entry=dict(task=task,valid_rows=len(q),exact=int(mask.sum()),new=[])
        for j,name in enumerate(('parent','neutral','tautomer')):
            known=set(keys[d['mi'][source],j]);new=np.isin(keys[d['mi'][q],j],list(known))
            entry['new'].append(dict(identity=name,covered=int(new.sum()),additional=int((new&~mask).sum())))
        records.append(entry)
    report=dict(records=records,identities_sha256=v.b.sha(file),test_labels_used=False,selection_uses_labels=False)
    v.b.save(path/'coverage.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
