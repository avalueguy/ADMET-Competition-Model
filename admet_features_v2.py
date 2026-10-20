"""Deterministic molecule features; learned preprocessing is fitted in each split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog('rdApp.*')
DESCRIPTORS = [(n, f) for n, f in Descriptors.descList if n not in {'Ipc', 'BalabanJ'}]
GENERATORS = [GetMorganGenerator(radius=r, fpSize=2048) for r in (2, 3)]
SCHEMA = {'version': 3, 'rdkit': rdkit.__version__, 'radii': [2, 3],
          'bits_each': 2048, 'descriptors': [n for n, _ in DESCRIPTORS]}


def features(smiles, cache_dir=None):
    smiles = list(map(str, smiles))
    digest = hashlib.sha256(json.dumps([SCHEMA, smiles], sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / (digest + '.npz') if cache_dir else None
    if path and path.exists():
        with np.load(path, allow_pickle=False) as z:
            return z['x'], z['canonical'].tolist(), z['scaffold'].tolist()
    x = np.zeros((len(smiles), 4096 + len(DESCRIPTORS)), np.float32)
    canonical, scaffold = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            canonical.append('INVALID:' + smi)
            scaffold.append('INVALID:' + smi)
            x[i, 4096:] = np.nan
            continue
        key = Chem.MolToSmiles(mol, canonical=True)
        canonical.append(key)
        # Keep acyclic molecules separated instead of one giant empty group.
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffold.append(scaf or 'ACYCLIC:' + key)
        for j, gen in enumerate(GENERATORS):
            x[i, j*2048:(j+1)*2048] = gen.GetFingerprintAsNumPy(mol)
        for j, (_, fn) in enumerate(DESCRIPTORS):
            try:
                v = float(fn(mol))
            except Exception:
                v = np.nan
            x[i, 4096+j] = v if np.isfinite(v) and abs(v) < 1e12 else np.nan
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=x, canonical=np.asarray(canonical), scaffold=np.asarray(scaffold))
    return x, canonical, scaffold


class DescriptorPreprocessor:
    def fit(self, x):
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.scaler = StandardScaler()
        self.scaler.fit(self.imputer.fit_transform(x[:, 4096:]))
        return self

    def transform(self, x):
        d = self.scaler.transform(self.imputer.transform(x[:, 4096:]))
        # Extremely heavy-tailed descriptors can otherwise destabilize a neural head.
        return np.concatenate([x[:, :4096], np.clip(d, -10, 10)], axis=1).astype(np.float32)
