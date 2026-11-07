"""Adapt the independent local-only fusion verifier to quantile forests."""
from pathlib import Path

p = Path(__file__).resolve().parent
s = (p / 'finalize_v176_local.py').read_text()
s = s.replace('v176', 'v182').replace('v175', 'v181')
s = s.replace('shared_graph_batch_20261027', 'quantile_forest_20261027')
s = s.replace('shared_graph_hybrid', 'quantile_forest')
s = s.replace("        assert check['global_scaffold_exclusion']", "        assert check['query_label_perturbation_invariant']")
s = s.replace("        model_folder = ROOT / 'admet_model_outputs_v181/models' / check['part'] / 'h96_l3_s5000'",
              "        model_folder = ROOT / 'admet_model_outputs_v181/models' / check['task'] / check['part'] / ('leaf' + str(selected[check['task']]['config']['leaf']))")
s = s.replace("    saved = ROOT / 'admet_model_outputs_v181/models/final/h96_l3_s5000/predictions.npz'", '')
s = s.replace("        p = array(saved, 'p' + str(k))",
              "        saved = ROOT / 'admet_model_outputs_v181/models' / task / 'final' / ('leaf' + str(record['config']['leaf'])) / 'predictions.npz'\n        p = array(saved, 'p')")
s = s.replace("array(saved, 'query' + str(k))", "array(saved, 'query')")
(p / 'finalize_v182_local.py').write_text(s)
