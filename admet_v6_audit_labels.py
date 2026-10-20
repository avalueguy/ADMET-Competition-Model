"""Training-label quality audit; no blind targets are used."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from rdkit import Chem

rows=[]
for folder in sorted(Path('admet_competition_data/data_benchmark').iterdir()):
    if not folder.is_dir():continue
    train=pd.read_csv(folder/'train.csv');valid=pd.read_csv(folder/'valid.csv')
    def key(s):
        mol=Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(mol) if mol is not None else 'INVALID:'+str(s)
    train['canonical']=train.Drug.map(key);valid['canonical']=valid.Drug.map(key)
    assert not set(train.canonical)&set(valid.canonical)
    y=train.Y.to_numpy(float);group=train.groupby('canonical').Y
    counts=group.size();ranges=group.max()-group.min()
    row=dict(task=folder.name,records=len(train),unique_molecules=len(counts),
             repeat_molecules=int((counts>1).sum()),conflicting_molecules=int((ranges>1e-8).sum()),
             largest_repeat=int(counts.max()),label_min=float(y.min()),label_max=float(y.max()),
             label_median=float(np.median(y)),label_q99=float(np.quantile(y,.99)))
    if not set(np.unique(y)).issubset({0,1}):
        # Per-row evaluation means a median at each duplicated molecule minimizes MAE.
        median=train.canonical.map(group.median())
        row['within_molecule_median_mae_lower_bound']=float(np.mean(np.abs(y-median)))
    rows.append(row)
pd.DataFrame(rows).to_csv('admet_diagnostics/v6_label_quality.csv',index=False)
print(json.dumps(rows))
