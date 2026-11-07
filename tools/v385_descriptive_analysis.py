"""Read completed development and confirmation only; never change selection."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 v384=json.loads((ROOT/'docs/v384_learning_curves.json').read_text());dev=json.loads((ROOT/'experiments/regression9_v385/outputs/development.json').read_text())
 records=[]
 for r in dev['records']:
  t=r['task'];transforms=[]
  for mode in('native','asinh','weighted_asinh','gaussian_rank','bounded_logit'):
   choices=[x for x in r['choices']if x['config']['mode']==mode]
   if not choices:continue
   b=max(choices,key=lambda x:(x['objective'],-x['weight']))
   transforms.append(dict(mode=mode,config=b['config'],weight=b['weight'],objective=b['objective'],MAE_gains=[x['gain']for x in b['parts']],rank_gains=[x.get('rank_gain')for x in b['parts']]))
  records.append(dict(task=t,frozen_config=r['best']['config'],transforms=transforms))
 labels=np.load(ROOT/'experiments/regression9_v385/official_rows.npz');manifest=json.loads((ROOT/'experiments/regression9_v385/data_manifest.json').read_text());label_summary=[]
 for t in manifest['pilot_tasks']:
  a=np.load(ROOT/'experiments/regression9_v385/controls'/t/'final.npz');y=labels['y'][a['fit']]
  label_summary.append(dict(task=t,fit_rows=len(y),quantiles=np.quantile(y,[0,.01,.1,.5,.9,.99,1.]).tolist(),ppbr_100_fraction=float((y==100).mean())if t=='ppbr_az'else None))
 value=dict(scope='Post-completion descriptive development/fit-label diagnostics, not additional confirmation tuning or an independent holdout.',
  frozen_selection_unchanged=True,fit_performed=False,records=records,label_summary=label_summary,recursive_metric_ablations=v384['ablations'])
 (ROOT/'docs/v385_descriptive_analysis.json').write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps(dict(status='COMPLETE',tasks=len(records),fit_performed=False)))
if __name__=='__main__':main()
