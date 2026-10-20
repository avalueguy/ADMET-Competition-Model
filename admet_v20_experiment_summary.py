"""Summarize completed experiments without modifying predictions or labels."""
import json
from pathlib import Path
import pandas as pd
from admet_v20_residual import task_names, anchor_valid, metric_for, utility, save_json

ROOT = Path(__file__).resolve().parent


def main():
    rare, direct, residual, mtl = [], [], [], []
    for name, output in [('admet_model_outputs_v20_rare', rare), ('admet_model_outputs_v20_map', direct)]:
        for p in sorted((ROOT/name).glob('*/decision.json')):
            route = json.loads(p.read_text())
            output.append(dict(task=p.parent.name, selection=route['selection']))
    for p in sorted((ROOT/'admet_model_outputs_v20').glob('*/decision.json')):
        route = json.loads(p.read_text())
        residual.append(dict(task=p.parent.name, alpha=route['selection']['alpha']))
    for task in task_names():
        y, base = anchor_valid(task)
        path = ROOT/'admet_mtl_outputs/valid'/f'{task}.csv'
        q = pd.read_csv(path).Y_pred.to_numpy(float)
        assert len(q)==len(y)
        metric = metric_for(task)
        before, after = utility(metric,y,base), utility(metric,y,q)
        mtl.append(dict(task=task, metric=metric, utility_before=before, utility_after=after,
                        utility_gain=after-before))
    result=dict(rare=rare,direct_map=direct,residual_historical_decisions=residual,mtl_valid=mtl,
                counts=dict(rare_trained=len(rare),rare_selected=sum(r['selection']['weight']>0 for r in rare),
                            map_trained=len(direct),map_selected=sum(r['selection']['weight']>0 for r in direct),
                            residual_tasks=len(residual),residual_selected=sum(r['alpha']>0 for r in residual),
                            mtl_tasks=len(mtl),mtl_improved=sum(r['utility_gain']>0 for r in mtl)),
                residual_caveat='Historical decisions preserved. Residual branch was not rerun after foldwise rank-anchor correction; do not treat it as corrected comparative evidence for v19-changed tasks.',
                score_forecast=None,platform_score=None,local_gpu_training=False,automatic_submission=False)
    save_json(ROOT/'admet_diagnostics/v20_experiment_summary.json',result)
    print(json.dumps(result['counts']),flush=True)


if __name__=='__main__':
    main()
