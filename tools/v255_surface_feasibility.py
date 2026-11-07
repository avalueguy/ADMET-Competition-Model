"""Label-free feasibility of solvent-exposed polarity beyond global 3D shape.

This is a diagnostic, not a fitted predictor or a submission. No assay label is
parsed. Six fixed conformers describe exposure variation rather than claiming
measured pH, permeability, binding affinity, or a physiological conformation.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig, RDLogger, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures, Descriptors, rdFreeSASA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import admet_v67_data as parser

NAMES = ('sasa', 'polar_sasa', 'apolar_sasa', 'donor_sasa', 'acceptor_sasa',
         'positive_charge_sasa', 'negative_charge_sasa', 'polar_fraction',
         'polar_burial_fraction', 'donor_acceptor_contact_fraction',
         'minimum_donor_acceptor_distance', 'centroid_charge_moment',
         'absolute_charge_sasa')
FACTORY = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exposure(smiles):
    mol, parse_mode = parser.parse_molecule(smiles)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    # Canonical atom order and structure-derived seed make input row order irrelevant.
    mol, _ = parser.parse_molecule(canonical)
    heavy = mol.GetNumHeavyAtoms()
    if heavy > 80:
        return dict(status='OVER80HEAVY', heavy=heavy, parse_mode=parse_mode)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(hashlib.sha256(canonical.encode()).hexdigest()[:8], 16) % 2147483647
    params.numThreads = 1
    params.maxIterations = 100
    params.pruneRmsThresh = .35
    params.timeout = 8
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=6, params=params))
    if not ids:
        return dict(status='NO_CONFORMER', heavy=heavy, parse_mode=parse_mode)
    if AllChem.MMFFHasAllMoleculeParams(mol):
        method = 'MMFF94'
        optimized = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=150)
    elif AllChem.UFFHasAllMoleculeParams(mol):
        method = 'UFF'
        optimized = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=150)
    else:
        return dict(status='NO_FORCEFIELD_PARAMETERS', heavy=heavy, parse_mode=parse_mode)
    AllChem.ComputeGasteigerCharges(mol, nIter=12, throwOnParamFailure=True)
    charges = np.array([float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms()])
    assert np.isfinite(charges).all()
    radii = np.array([Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()])
    polar = np.array([a.GetAtomicNum() in (7, 8, 16) or
                      (a.GetAtomicNum() == 1 and a.GetNeighbors()[0].GetAtomicNum() in (7, 8, 16))
                      for a in mol.GetAtoms()])
    donors, acceptors = set(), set()
    for f in FACTORY.GetFeaturesForMol(mol):
        if f.GetFamily() == 'Donor':
            donors.update(f.GetAtomIds())
        elif f.GetFamily() == 'Acceptor':
            acceptors.update(f.GetAtomIds())
    donor_surface = set(donors)
    for i in donors:
        donor_surface.update(a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetAtomicNum() == 1)
    graph = Chem.GetDistanceMatrix(mol)
    pairs = [(i, j) for i in sorted(donors) for j in sorted(acceptors) if graph[i, j] >= 4]
    isolated_polar = np.sum(4 * np.pi * (radii[polar] + 1.4) ** 2)
    values, energies, converged = [], [], []
    for cid, (status, energy) in zip(ids, optimized):
        rdFreeSASA.CalcSASA(mol, radii.tolist(), confIdx=cid)
        sasa = np.array([float(a.GetProp('SASA')) for a in mol.GetAtoms()])
        xyz = np.array(mol.GetConformer(cid).GetPositions())
        total = sasa.sum()
        distances = np.array([np.linalg.norm(xyz[i] - xyz[j]) for i, j in pairs])
        polar_area = sasa[polar].sum()
        # A distance contact is a descriptor; it is not proof of a hydrogen bond.
        row = [total, polar_area, sasa[~polar].sum(), sasa[list(donor_surface)].sum(),
               sasa[list(acceptors)].sum(), np.dot(sasa, np.maximum(charges, 0)),
               np.dot(sasa, np.maximum(-charges, 0)), polar_area / max(total, 1e-12),
               1 - polar_area / max(isolated_polar, 1e-12),
               float(np.mean(distances < 3.5)) if len(distances) else 0.,
               float(distances.min()) if len(distances) else 20.,
               np.linalg.norm(np.sum(charges[:, None] * (xyz - xyz.mean(0)), axis=0)),
               np.dot(sasa, np.abs(charges))]
        values.append(row)
        energies.append(energy)
        converged.append(status == 0)
    x = np.array(values)
    assert x.shape == (len(ids), len(NAMES)) and np.isfinite(x).all()
    energies = np.array(energies)
    weights = np.exp(-(energies - energies.min()).clip(0, 50) / .593)
    weights /= weights.sum()
    mean = weights @ x
    sd = np.sqrt(weights @ ((x - mean) ** 2))
    return dict(status='PASS', heavy=heavy, parse_mode=parse_mode, forcefield=method,
                conformers=len(ids), convergence_fraction=float(np.mean(converged)),
                energy_range=float(np.ptp(energies)), mean=mean.tolist(),
                sd=sd.tolist(), minimum=x.min(0).tolist(), maximum=x.max(0).tolist(),
                tpsa=float(Descriptors.TPSA(Chem.RemoveHs(mol))),
                mol_logp=float(Descriptors.MolLogP(Chem.RemoveHs(mol))))


def main():
    RDLogger.DisableLog('rdApp.*')
    path = ROOT / 'docs/v255_surface_feasibility.json'
    assert not path.exists(), 'Preserve diagnostic; do not rerun blindly'
    source = ROOT / 'admet_competition_data/data_benchmark/caco2_wang/train.csv'
    # Read Drug only; label values and valid/test files are never examined.
    structures = sorted({r['Drug'] for r in parser.table(source)})
    ranked = sorted(structures, key=lambda s: (parser.parse_molecule(s)[0].GetNumHeavyAtoms(), s))
    selected = [ranked[i] for i in np.unique(np.linspace(0, len(ranked) - 1, 24).astype(int))]
    started = time.monotonic()
    records = []
    for sm in selected:
        before = time.monotonic()
        try:
            result = exposure(sm)
        except Exception as exc:
            result = dict(status='FAILED_FEATURE', error=type(exc).__name__ + ': ' + str(exc))
        records.append(dict(structure_sha256=hashlib.sha256(sm.encode()).hexdigest(),
                            seconds=time.monotonic() - before, **result))
        print(json.dumps(dict(done=len(records), status=result['status'], seconds=records[-1]['seconds'])), flush=True)
    # Repeat a synthetic structure in a different notation, without assay labels.
    a, z = exposure('CC(O)N'), exposure('NC(O)C')
    for key in ('mean', 'sd', 'minimum', 'maximum'):
        np.testing.assert_array_equal(a[key], z[key])
    report = dict(status='LABEL_FREE_FEASIBILITY_COMPLETE', source_sha256=sha(__file__),
                  input_sha256=sha(source), rdkit=rdBase.rdkitVersion, selected_by='24 evenly spaced heavy-atom-count ranks from unique Caco2 train SMILES',
                  assay_labels_accessed=False, valid_test_files_accessed=False, model_fitted=False,
                  names=list(NAMES), records=records, elapsed_seconds=time.monotonic() - started,
                  notation_invariance=True, prediction_gain_evaluated=False,
                  limitations='Forcefield conformers and partial charges approximate exposure. No assay pH or measured ionization. This audit cannot establish predictive gain.')
    parser.save(path, report)
    print(json.dumps(dict(status=report['status'], pass_count=sum(r['status']=='PASS' for r in records), total=len(records))), flush=True)


if __name__ == '__main__':
    main()
