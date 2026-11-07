"""Separate MAE-only research signals from deployable official-metric updates."""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--version',choices=['377','378','379','380','381'],required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];v='v'+args.version
    exp=root/f'experiments/regression9_{v}'
    conf=json.loads((exp/'outputs/confirmation.json').read_text())
    delivery=json.loads((root/f'docs/{v}_delivery.json').read_text())
    records=[]
    for r in conf['records']:
        parts=r['confirmation'];mae_positive=all(x['gain']>0 for x in parts)
        if not mae_positive:continue
        rank_positive=all(x.get('rank_gain',1)>0 for x in parts)
        records.append(dict(task=r['task'],mae_relative_gains=[x['gain'] for x in parts],
                            mean_gain=float(np.mean([x['gain'] for x in parts])),
                            rank_gains=[x.get('rank_gain') for x in parts],
                            official_metric_all_parts_positive=rank_positive,
                            accepted_for_submission=r['experimental_eligible'],
                            calibrated_rank_reference='rank_gain' in parts[0]))
    mean={p:sum(next(x['gain'] for x in r['confirmation'] if x['part']==p)
                for r in conf['records'] if all(x['gain']>0 for x in r['confirmation']))/9
          for p in ('fold0','fold1','fold2','valid')}
    value=dict(version=v,mae_only_all_four_positive_tasks=records,
               research_only_mean_over_nine_with_others_zero=mean,
               deployable_mean_over_nine=delivery['nine_task_mean_relative_MAE_gain_by_part'],
               caution='MAE-only signals on four Spearman tasks may lose official points; calibrated reference is not original-model native MAE. This analysis creates no CSV and changes no selection.')
    (root/f'docs/{v}_research_vs_deployment.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value))


if __name__=='__main__':main()
