"""Leakage and submission invariants independent of model implementation."""
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import admet_v67_data as b
from verify_submission import verify


class IntegrityTests(unittest.TestCase):
    def test_official_valence_exception_preserves_graph(self):
        smiles = "CC1=CC=C[NH+2]([O-])[CH-]1"
        mol, mode = b.parse_molecule(smiles)
        self.assertEqual(mol.GetNumAtoms(), 8)
        self.assertEqual(b.Chem.MolToSmiles(mol, isomericSmiles=False), smiles)
        self.assertIn("original_graph_preserved", mode)

    def test_query_outlier_cannot_change_preprocessing(self):
        x = np.zeros((3, 4098), np.float32)
        x[:, 4096:] = [[1, 3], [3, 5], [1e8, -1e8]]
        d = {"x": x, "mi": np.array([0, 1, 2])}
        first = b.preprocess(d, np.array([0, 1]))
        d["x"][2, 4096:] = [-1e20, 1e20]
        second = b.preprocess(d, np.array([0, 1]))
        for key in ("median", "mean", "scale"):
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first["mean"], [2, 4])

    def test_cross_task_shared_scaffold_is_excluded(self):
        d = dict(split=np.array([0, 0, 0]), rg=np.array(["benzene", "benzene", "pyridine"]),
                 mi=np.array([0, 1, 2]), canonical=np.array(["A", "B", "C"]))
        with self.assertRaises(AssertionError):
            b.assert_excluded(d, np.array([0]), np.array([1]))
        b.assert_excluded(d, np.array([0]), np.array([2]))

    def test_canonical_identity_cannot_cross_group_boundaries(self):
        d = dict(split=np.array([0, 0]), rg=np.array(["old1", "old2"]),
                 mi=np.array([0, 1]), canonical=np.array(["same", "same"]))
        with self.assertRaises(AssertionError):
            b.assert_excluded(d, np.array([0]), np.array([1]))

    def test_test_rows_can_never_be_fit_rows(self):
        d = dict(split=np.array([2, 0]), rg=np.array(["a", "b"]),
                 mi=np.array([0, 1]), canonical=np.array(["A", "B"]))
        with self.assertRaises(AssertionError):
            b.assert_excluded(d, np.array([0]), np.array([1]))

    def test_submission_rejects_numeric_id_reformatting(self):
        header = ["task", "sample_idx", "drug_id", "smiles", "y_pred", "metric"]
        good = [header, ["task", "0", "000123", "C", "0.4", "AUROC"]]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            def write(path, rows):
                stream = io.StringIO(newline="")
                csv.writer(stream, lineterminator="\r\n").writerows(rows)
                path.write_bytes(b"\xef\xbb\xbf"+stream.getvalue().encode())
            template, file = directory/"template.csv", directory/"pred.csv"
            write(template, good); write(file, good)
            self.assertEqual(verify(file, template)["status"], "PASS")
            bad = [header, ["task", "0", "123", "C", "0.4", "AUROC"]]
            write(file, bad)
            with self.assertRaises(AssertionError):
                verify(file, template)


if __name__ == "__main__":
    unittest.main(verbosity=2)
