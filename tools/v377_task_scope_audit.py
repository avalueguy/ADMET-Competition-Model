"""Frozen task counts and baseline errors; read-only arithmetic, no fitting."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def main():
    exp = ROOT/'experiments/regression9_v377'
    data = dict(np.load(exp/'official_rows.npz'))
    manifest = json.loads((exp/'data_manifest.json').read_text(encoding='utf-8'))
    card = json.loads((ROOT/'docs/platform_scorecard_v374_20261105.json').read_text(encoding='utf-8'))
    feedback = {r['task']: r for r in card['tasks']}
    rows = []
    for task in manifest['pilot_tasks']:
        metrics = feedback[task]
        key = manifest['tasks'].index(task)
        counts = {name: int(((data['tid'] == key) & (data['split'] == i)).sum())
                  for i, name in enumerate(('train', 'valid', 'test'))}
        parts = []
        for part in ('fold0','fold1','fold2','valid'):
            a = dict(np.load(exp/'controls'/task/(part+'.npz')))
            y = data['y'][a['query']]
            parts.append(dict(part=part, fit_rows=len(a['fit']), query_rows=len(y),
                              groups=len(np.unique(data['groups'][data['mi'][a['query']]])),
                              reference_MAE=float(np.abs(y-a['p']).mean()),
                              reference_spearman=float(spearmanr(y,a['score_p']).statistic)))
        rows.append(dict(task=task, official_metric=metrics['metric'], official_raw=metrics['raw'],
                         counts=counts, parts=parts,
                         five_percent_official_MAE_target=metrics['raw']*.95 if metrics['metric']=='MAE' else None,
                         native_reference_is_calibrated_rank=metrics['metric']=='Spearman'))
    value=dict(status='PASS', rows=rows, no_fitting=True,
               test_labels_used=False, warning='Historical OOF is partly proxy. Four Spearman task MAEs use explicitly calibrated rank references; not exact old-model native error.')
    (ROOT/'docs/v377_task_scope_audit.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([dict(task=r['task'],metric=r['official_metric'],counts=r['counts'],
                           reference_MAE=[p['reference_MAE'] for p in r['parts']]) for r in rows]))


if __name__=='__main__':main()
