"""Label-free deterministic official-structure conformer descriptors."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from rdkit import Chem,RDLogger,rdBase
from rdkit.Chem import AllChem,rdMolDescriptors as md
from rdkit.Chem.MolStandardize import rdMolStandardize
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v95_graph_specialists as v
OUT=ROOT/'admet_model_outputs_v98/features'
NAMES=('asphericity','eccentricity','inertial_shape','radius_gyration','spherocity','pmi1','pmi2','pmi3','npr1','npr2','pbf')
DIM=11+80+210+224+114

def sig():return hashlib.sha256((v.b.sha(__file__)+v.sig()).encode()).hexdigest()

def build_one(item):
    index,smiles=item;RDLogger.DisableLog('rdApp.*')
    fail=np.full(2*DIM+5,np.nan,np.float32)
    try:
        mol=v.old_graph.chemistry.parse_molecule(smiles)[0]
        mol=rdMolStandardize.LargestFragmentChooser().choose(mol)
        heavy=mol.GetNumHeavyAtoms()
        if heavy>100:return index,fail,'over100heavy'
        mol=Chem.AddHs(mol)
        params=AllChem.ETKDGv3();params.randomSeed=984026+index;params.numThreads=1
        params.maxIterations=75;params.pruneRmsThresh=.5;params.timeout=8
        ids=list(AllChem.EmbedMultipleConfs(mol,numConfs=3,params=params))
        if not ids:return index,fail,'embedding_failed'
        energies=[];optimized=[]
        if AllChem.MMFFHasAllMoleculeParams(mol):
            result=AllChem.MMFFOptimizeMoleculeConfs(mol,numThreads=1,maxIters=120)
        elif AllChem.UFFHasAllMoleculeParams(mol):
            result=AllChem.UFFOptimizeMoleculeConfs(mol,numThreads=1,maxIters=120)
        else:result=[(-1,0.) for _ in ids]
        mol=Chem.RemoveHs(mol);vectors=[]
        scalar=(md.CalcAsphericity,md.CalcEccentricity,md.CalcInertialShapeFactor,md.CalcRadiusOfGyration,
                md.CalcSpherocityIndex,md.CalcPMI1,md.CalcPMI2,md.CalcPMI3,md.CalcNPR1,md.CalcNPR2,md.CalcPBF)
        for cid,(status,energy) in zip(ids,result):
            values=[fn(mol,confId=cid) for fn in scalar]
            values+=list(md.CalcAUTOCORR3D(mol,confId=cid))
            values+=list(md.CalcRDF(mol,confId=cid))
            values+=list(md.CalcMORSE(mol,confId=cid))
            values+=list(md.CalcWHIM(mol,confId=cid))
            vec=np.array(values,float);assert len(vec)==DIM
            vectors.append(vec);energies.append(energy);optimized.append(status==0)
        vv=np.array(vectors);finite=np.isfinite(vv);vv=np.where(finite,vv,np.nan)
        with np.errstate(all='ignore'):
            mean=np.nanmean(vv,axis=0);sd=np.nanstd(vv,axis=0)
        tail=[len(ids),float(np.mean(optimized)),float(np.max(energies)-np.min(energies)),heavy,1.]
        out=np.r_[mean,sd,tail].astype(np.float32);out[~np.isfinite(out)]=np.nan
        return index,out,'success'
    except Exception as e:
        return index,fail,type(e).__name__+':'+str(e)[:100]

def main():
    OUT.mkdir(parents=True,exist_ok=True);d=v.load();smiles=[None]*len(d['views']['physchem']);cursor=0
    for task in d['tasks']:
        for part in ('train','valid','test'):
            rows=([{'Drug':r['smiles']} for r in d['template'] if r['task']==task] if part=='test'
                else v.b.table(ROOT/'admet_competition_data/data_benchmark'/task/(part+'.csv')))
            for row in rows:
                i=d['mi'][cursor];assert smiles[i] in (None,row['Drug']);smiles[i]=row['Drug'];cursor+=1
    smhash=hashlib.sha256(json.dumps(smiles).encode()).hexdigest()
    assert smhash==json.loads((ROOT/'admet_model_outputs_v67/features/official_chemistry.json').read_text())['smiles_sha256']
    x=np.full((len(smiles),2*DIM+5),np.nan,np.float32);status=np.full(len(smiles),'pending',dtype='<U130');started=time.monotonic()
    file=OUT/'conformers.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();print('verified existing',flush=True);return
    chunks=OUT/'chunks';chunks.mkdir(exist_ok=True);jobs=[]
    for lo in range(0,len(smiles),1024):
        p=chunks/f'{lo}.npz'
        if p.exists():
            z=np.load(p);assert str(z['signature'])==sig();x[lo:lo+len(z['x'])]=z['x'];status[lo:lo+len(z['status'])]=z['status']
        else:jobs.append(lo)
    with ProcessPoolExecutor(max_workers=12) as pool:
        for lo in jobs:
            hi=min(lo+1024,len(smiles));results=list(pool.map(build_one,enumerate(smiles[lo:hi],lo),chunksize=4))
            for i,row,state in results:x[i]=row;status[i]=state
            np.savez_compressed(chunks/f'{lo}.npz',signature=sig(),x=x[lo:hi],status=status[lo:hi])
            v.emit(stage='conformers',done=hi,total=len(smiles),seconds=round(time.monotonic()-started),success=int(np.sum(status=='success')))
    # Explicit validity column remains finite for all molecules.
    x[:,-1]=status=='success'
    np.savez_compressed(file,signature=sig(),x=x,status=status,smiles_sha256=smhash)
    counts={s:int(np.sum(status==s)) for s in np.unique(status)}
    v.b.save(OUT/'audit.json',dict(status='COMPLETE',signature=sig(),sha256=v.b.sha(file),rdkit=rdBase.rdkitVersion,
        molecules=len(smiles),dimensions=x.shape[1],outcomes=counts,test_labels_used=False,external_data=False,pretrained_weights=False,
        conformers=3,seed='984026 + immutable molecule index',method='ETKDGv3; MMFF/UFF 120 steps; heavy-atom descriptors mean/std',
        scalar_names=NAMES,vector_sizes=dict(autocorr3d=80,rdf=210,morse=224,whim=114)))

if __name__=='__main__':main()
