"""Versioned label-free features. Training is performed remotely by the runner."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetMorganFeatureAtomInvGen, GetRDKitFPGenerator
from admet_v5_legacy_mtl import existing_features
from admet_features_v2 import SCHEMA

RDLogger.DisableLog('rdApp.*')
DESC_NAMES = [n for n in SCHEMA['descriptors'] if n not in {'AvgIpc', 'Ipc', 'BalabanJ'}]
DESC_IX = [4096 + SCHEMA['descriptors'].index(n) for n in DESC_NAMES]
FEATURE_SCHEMA = {'version': 6, 'rdkit': rdBase.rdkitVersion, 'descriptors': DESC_NAMES,
                  'fingerprints': ['Morgan2_count2048_chiral', 'FCFP3_count2048', 'RDKit2048']}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_task(task):
    import pandas as pd
    root = Path('admet_competition_data/data_benchmark') / task
    frames = {s: pd.read_csv(root / (s + '.csv')) for s in ['train', 'valid', 'test']}
    smiles = [str(s) for f in frames.values() for s in f.Drug]
    raw, canonical, groups = existing_features(smiles)
    key = hashlib.sha256(json.dumps([FEATURE_SCHEMA, smiles], sort_keys=True).encode()).hexdigest()
    path = Path('admet_model_outputs_v6/features') / (task + '.npz')
    if path.exists():
        with np.load(path) as z:
            assert str(z['key']) == key
            count, fcfp, rdkit = z['count'], z['fcfp'], z['rdkit']
    else:
        gens = [GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True),
                GetMorganGenerator(radius=3, fpSize=2048, atomInvariantsGenerator=GetMorganFeatureAtomInvGen()),
                GetRDKitFPGenerator(fpSize=2048)]
        count = np.zeros((len(smiles), 2048), np.float32)
        fcfp = np.zeros_like(count); rdkit = np.zeros_like(count)
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            count[i] = gens[0].GetCountFingerprintAsNumPy(mol)
            fcfp[i] = gens[1].GetCountFingerprintAsNumPy(mol)
            rdkit[i] = gens[2].GetFingerprintAsNumPy(mol)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + '.' + str(os.getpid()) + '.npz')
        np.savez_compressed(temporary, count=count, fcfp=fcfp, rdkit=rdkit, key=key)
        os.replace(temporary, path)
    with np.load(Path('admet_pretrained/embeddings') / (task + '.npz')) as z:
        assert str(z['digest']) == hashlib.sha256(json.dumps(smiles).encode()).hexdigest()
        emb = z['x'].copy()
    desc = raw[:, DESC_IX].copy()
    return dict(frames=frames, smiles=smiles, raw=raw, canonical=np.asarray(canonical),
                groups=np.asarray(groups), desc=desc, emb=emb, count=count, fcfp=fcfp, rdkit=rdkit,
                feature_sha256=sha(path))


def dense(data):
    return np.concatenate([np.log1p(data['count']), np.log1p(data['fcfp']), data['desc'], data['emb']], axis=1)


def mixed_kernel(data):
    """A convex mixture of three PSD generalized Tanimoto kernels."""
    import torch
    n = len(data['count']); result = np.zeros((n, n), np.float32)
    for weight, arr in [(0.5, data['count']), (0.25, data['fcfp']), (0.25, data['rdkit'])]:
        xx = torch.as_tensor(arr, device='cuda', dtype=torch.float32)
        norm = (xx * xx).sum(1)
        for start in range(0, n, 256):
            dot = xx[start:start+256] @ xx.T
            den = norm[start:start+256, None] + norm[None, :] - dot
            kk = dot / den.clamp_min(1e-8)
            result[start:start+256] += weight * kk.cpu().numpy()
        del xx
    return result


def onehot(v, choices):
    return [float(v == c) for c in choices] + [float(v not in choices)]


def make_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        mol = Chem.MolFromSmiles('[He]')
    atom = []
    for a in mol.GetAtoms():
        atom.append(onehot(a.GetAtomicNum(), [1,5,6,7,8,9,14,15,16,17,35,53]) +
                    onehot(a.GetTotalDegree(), [0,1,2,3,4,5]) + onehot(a.GetFormalCharge(), [-2,-1,0,1,2]) +
                    onehot(a.GetTotalNumHs(), [0,1,2,3,4]) + onehot(int(a.GetChiralTag()), [0,1,2,3]) +
                    onehot(int(a.GetHybridization()), [1,2,3,4,5,6]) +
                    [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass()/100.])
    src, dst, rev, bond = [], [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = onehot(b.GetBondTypeAsDouble(), [1.,1.5,2.,3.]) + [float(b.GetIsConjugated()), float(b.IsInRing())] + onehot(int(b.GetStereo()), [0,1,2,3,4,5])
        edge = len(src); src.extend([i,j]); dst.extend([j,i]); rev.extend([edge+1,edge]); bond.extend([bf,bf])
    return (np.asarray(atom, np.float32), np.asarray(bond, np.float32).reshape(-1, 14),
            np.asarray(src, np.int64), np.asarray(dst, np.int64), np.asarray(rev, np.int64))


def graph_batch(graphs, device='cuda'):
    import torch
    aa, bb, ss, dd, rr, gg = [], [], [], [], [], []
    atoms = edges = 0
    for k, (a, b, s, d, r) in enumerate(graphs):
        aa.append(a); bb.append(b); ss.append(s+atoms); dd.append(d+atoms); rr.append(r+edges)
        gg.append(np.full(len(a), k, np.int64)); atoms += len(a); edges += len(b)
    arrays = [np.concatenate(v) for v in [aa,bb,ss,dd,rr,gg]]
    return [torch.as_tensor(v, device=device) for v in arrays]
