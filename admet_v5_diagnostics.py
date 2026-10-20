"""Summarize completed experiments; conditional uncertainty is not a score forecast."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from admet_features_v2 import features
from admet_upgrade_v3 import metrics, utility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bootstrap', action='store_true')
    args = ap.parse_args()
    root = Path('admet_model_outputs_v5')
    out = Path('admet_diagnostics'); out.mkdir(exist_ok=True)
    rows = []
    for p in sorted(root.glob('*/result.json')):
        r = json.loads(p.read_text())
        task = r['task']
        before, after = r['valid_before'], r['valid_blend']
        confirm = Path('admet_v5_confirmation')/task/'confirmation.json'
        c = json.loads(confirm.read_text()) if confirm.exists() else {}
        row = dict(task=task, family=r['chosen'], metric=r['metric'],
                   provisional_valid_pass=r['provisional_valid_pass'],
                   matched_confirmation=c.get('accepted',None),
                   official_metric_gain=utility(r['metric'],after)-utility(r['metric'],before),
                   **{'before_'+k:v for k,v in before.items()},
                   **{'after_'+k:v for k,v in after.items()})
        if args.bootstrap:
            df = pd.read_csv(p.parent/'validation_comparison.csv')
            smiles = pd.read_csv(Path('admet_competition_data/data_benchmark')/task/'valid.csv').Drug.tolist()
            _, _, groups = features(smiles, root/'features')
            unique = sorted(set(groups))
            if len(unique)<2:
                row.update(conditional_ci_low=None,conditional_ci_high=None,
                           bootstrap_valid_samples=0,valid_scaffolds=len(unique),valid_rows=len(df),
                           uncertainty_note='Fewer than 2 scaffold clusters; cluster bootstrap is uninformative')
                rows.append(row)
                continue
            ix = {g:np.flatnonzero(np.asarray(groups)==g) for g in unique}
            rng = np.random.default_rng(2030)
            y = df.Y_true.to_numpy(float); a = df.baseline.to_numpy(float); b = df.blend.to_numpy(float)
            diffs = []
            for _ in range(500):
                sample = np.concatenate([ix[g] for g in rng.choice(unique,len(unique),replace=True)])
                if r['kind']=='classification' and len(np.unique(y[sample]))<2:
                    continue
                diffs.append(utility(r['metric'],metrics(r['kind'],y[sample],b[sample]))-
                             utility(r['metric'],metrics(r['kind'],y[sample],a[sample])))
            row.update(conditional_ci_low=float(np.quantile(diffs,.025)),
                       conditional_ci_high=float(np.quantile(diffs,.975)),
                       bootstrap_valid_samples=len(diffs),valid_scaffolds=len(unique),valid_rows=len(y))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/('v5_uncertainty.csv' if args.bootstrap else 'v5_progress.csv'),index=False)
    print(json.dumps(dict(completed=len(rows),provisional=sum(r['provisional_valid_pass'] for r in rows),
                          confirmed=[r['task'] for r in rows if r['matched_confirmation']],
                          last_tasks=[r['task'] for r in rows[-3:]],platform_score=None)),flush=True)
    if args.bootstrap:
        (out/'v5_uncertainty_notes.txt').write_text(
            'Paired scaffold bootstrap on reused official valid after selection, seed=2030, 500 draws. '
            'Intervals condition on fixed predictions and ignore selection, retraining and distribution shift. '
            'With fewer than two distinct scaffolds no cluster interval is reported. '
            'They are descriptive and do not change the preregistered promotion gate. '
            'Gains use each task metric, with positive always better; they are not platform points.\n',encoding='utf8')


if __name__=='__main__':main()
