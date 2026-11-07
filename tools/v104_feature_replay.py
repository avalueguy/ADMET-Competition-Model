"""Independent sparse fingerprint regeneration on fixed official samples."""
import json,sys
from pathlib import Path
import numpy as np
from scipy import sparse
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator,GetMorganFeatureAtomInvGen
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v104_sparse_features as v
def main():
    d=v.b.load_data();rows=[];molecule_smiles={}
    for k,task in enumerate(d['tasks']):
        for si,part in enumerate(('train','valid','test')):
            strings=[r['smiles'] for r in d['template'] if r['task']==task] if si==2 else [r['Drug'] for r in v.b.table(ROOT/'admet_competition_data/data_benchmark'/task/(part+'.csv'))]
            ix=np.flatnonzero((d['tid']==k)&(d['split']==si));assert len(ix)==len(strings)
            for i,sm in zip(ix,strings):
                m=int(d['mi'][i]);assert m not in molecule_smiles or molecule_smiles[m]==sm;molecule_smiles[m]=sm
    ids=[]
    for si in (0,1,2):
        ix=np.flatnonzero(d['split']==si);ids.extend(d['mi'][ix[np.linspace(0,len(ix)-1,4,dtype=int)]].tolist())
    gens=(GetMorganGenerator(radius=2,includeChirality=True),GetMorganGenerator(radius=3,includeChirality=True,atomInvariantsGenerator=GetMorganFeatureAtomInvGen()))
    for j,gen in enumerate(gens):
        xx=sparse.load_npz(v.OUT/f'view{j}.npz');vocab=np.load(v.OUT/f'vocabulary{j}.npy')
        for m in sorted(set(ids)):
            mol,_=v.parser.parse_molecule(molecule_smiles[m]);expected={int(key):int(val) for key,val in gen.GetSparseCountFingerprint(mol).GetNonzeroElements().items()}
            r=xx.getrow(m);actual={int(vocab[c]):int(val) for c,val in zip(r.indices,r.data)};assert actual==expected
            rows.append(dict(view=j,molecule=m,substructures=len(actual)))
    v.b.save(v.OUT/'replay.json',dict(status='PASS',sample_molecules=len(set(ids)),records=rows,query_labels_used=False,source_smiles_mapping_reconstructed=True))
    print(json.dumps(dict(status='PASS',sample_molecules=len(set(ids)))),flush=True)
if __name__=='__main__':main()
