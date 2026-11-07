"""Unfolded molecular substructure IDs from official structures only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import hashlib,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import sparse
from rdkit import Chem,RDLogger,rdBase
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator,GetMorganFeatureAtomInvGen
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import admet_v77_ranking as b
import admet_v67_data as parser
OUT=ROOT/'admet_model_outputs_v104/features'

def sig():return hashlib.sha256((b.sha(__file__)+b.sha(parser.__file__)).encode()).hexdigest()
def one(sm):
    RDLogger.DisableLog('rdApp.*');mol,_=parser.parse_molecule(sm)
    gs=(GetMorganGenerator(radius=2,includeChirality=True),GetMorganGenerator(radius=3,includeChirality=True,atomInvariantsGenerator=GetMorganFeatureAtomInvGen()))
    return [g.GetSparseCountFingerprint(mol).GetNonzeroElements() for g in gs]

def build():
    d=b.load_data();smiles=list(dict.fromkeys(r['smiles'] for r in []))
    # Loader observation order is task train/valid/template; recover exact strings.
    smiles=[];seen=set()
    for task in d['tasks']:
        for part in ('train','valid'):
            for r in b.table(ROOT/'admet_competition_data/data_benchmark'/task/(part+'.csv')):
                if r['Drug'] not in seen:seen.add(r['Drug']);smiles.append(r['Drug'])
        for r in d['template']:
            if r['task']==task and r['smiles'] not in seen:seen.add(r['smiles']);smiles.append(r['smiles'])
    assert len(smiles)==len(d['views']['physchem']);OUT.mkdir(parents=True,exist_ok=True)
    schema=dict(signature=sig(),rdkit=rdBase.rdkitVersion,smiles_sha256=hashlib.sha256(json.dumps(smiles).encode()).hexdigest())
    if (OUT/'audit.json').exists():
        a=json.loads((OUT/'audit.json').read_text());assert a['schema']==schema
        for j in range(2):assert b.sha(OUT/f'view{j}.npz')==a['views'][j]['sha256']
        return
    arrays=[[] for _ in range(2)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,views in enumerate(pool.map(one,smiles,chunksize=128)):
            for j,z in enumerate(views):arrays[j].append(z)
            if i%10000==0:print(json.dumps(dict(stage='sparse_features',molecules=i,total=len(smiles))),flush=True)
    records=[]
    for j,rows in enumerate(arrays):
        vocab=np.array(sorted({key for r in rows for key in r}),dtype=np.uint64);mapping={int(x):i for i,x in enumerate(vocab)}
        ix=[];values=[];ptr=[0]
        for r in rows:
            for key,val in sorted(r.items()):ix.append(mapping[key]);values.append(val)
            ptr.append(len(ix))
        x=sparse.csr_matrix((np.array(values,np.float32),np.array(ix,np.int32),np.array(ptr,np.int64)),shape=(len(smiles),len(vocab)))
        sparse.save_npz(OUT/f'view{j}.npz',x);np.save(OUT/f'vocabulary{j}.npy',vocab)
        records.append(dict(sha256=b.sha(OUT/f'view{j}.npz'),vocabulary_sha256=b.sha(OUT/f'vocabulary{j}.npy'),shape=list(x.shape),nnz=int(x.nnz)))
    b.save(OUT/'audit.json',dict(status='PASS',schema=schema,views=records,test_labels_used=False,source='Official train/valid/template SMILES; no labels. Integer substructure IDs without folding; vocabulary is structure-only, no learned statistics.'))
    print(json.dumps(dict(stage='features_complete',views=records)),flush=True)
if __name__=='__main__':build()
