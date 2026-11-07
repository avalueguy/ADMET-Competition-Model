"""Independent actual-scaffold coverage and concentration audit, no retuning."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import sys,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from rdkit import Chem,RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import admet_v67_data as parser
OUT=ROOT/'admet_model_outputs_v152'
def one(sm):
    RDLogger.DisableLog('rdApp.*');mol,_=parser.parse_molecule(sm);can=Chem.MolToSmiles(mol,isomericSmiles=False)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False) or 'ACYCLIC:'+can
def main():
    d=v.load();smiles=[];seen=set()
    for t in d['tasks']:
        for part in ('train','valid'):
            for r in v.b.table(ROOT/'admet_competition_data/data_benchmark'/t/(part+'.csv')):
                if r['Drug'] not in seen:seen.add(r['Drug']);smiles.append(r['Drug'])
        for r in d['template']:
            if r['task']==t and r['smiles'] not in seen:seen.add(r['smiles']);smiles.append(r['smiles'])
    OUT.mkdir(exist_ok=True);file=OUT/'scaffolds.npz'
    if file.exists():true=np.load(file)['groups']
    else:
        with ProcessPoolExecutor(max_workers=4) as pool:true=np.array(list(pool.map(one,smiles,chunksize=128)))
        np.savez_compressed(file,groups=true,smiles_sha256=hashlib.sha256(json.dumps(smiles).encode()).hexdigest())
    records=[]
    for t in v.tasks(d):
        k=d['tasks'].index(t);parts=[]
        for part in ('fold0','fold1','fold2','valid','final'):
            fit,q=v.split(d,k,part);g=true[d['mi'][q]];_,ct=np.unique(g,return_counts=True)
            parts.append(dict(part=part,fit=len(fit),query=len(q),cached_groups=len(np.unique(d['groups'][q])),true_scaffolds=len(ct),largest_true_scaffold_rows=int(ct.max()),true_scaffold_overlap_with_fit=len(set(g)&set(true[d['mi'][fit]]))))
        records.append(dict(task=t,parts=parts))
    v.b.save(OUT/'audit.json',dict(status='COMPLETE',selection_effect='NONE',scaffold_sha256=v.b.sha(file),records=records,test_labels_used=False,
        note='Recomputes chemical scaffolds from structures independently of cached exclusion components. Does not alter historical splits or model selection.'))
    print(json.dumps(records),flush=True)
if __name__=='__main__':main()
