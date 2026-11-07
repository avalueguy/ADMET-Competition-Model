"""Fixed-choice raw-vs-calibration description; never reselect on confirmation."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    exp=ROOT/'experiments/regression9_v386'
    dev=json.loads((exp/'outputs/development.json').read_text(encoding='utf-8'))
    delivery=json.loads((ROOT/'docs/v386_delivery.json').read_text(encoding='utf-8'))
    records=[]
    for r in dev['records']:
        best=r['best'];config=best['config'];comparisons=[]
        for mode in('raw','mono02','mono10'):
            c=next(c for c in r['choices']if c['config']['base']==config['base']and c['config']['calibration']==mode and c['weight']==best['weight'])
            comparisons.append(dict(config=c['config'],weight=c['weight'],development_parts=c['parts'],objective=c['objective']))
        records.append(dict(task=r['task'],frozen_choice=config,weight=best['weight'],fixed_weight_development_comparison=comparisons,
            eligible=next(x['accepted']for x in delivery['task_evidence']if x['task']==r['task'])))
    result=dict(signature=dev['signature'],records=records,descriptive_only=True,no_confirmation_reselection=True,
        caveat='Multiple public partitions have long been reused; this describes the completed configurations and does not isolate a new blind-test causal effect.')
    (ROOT/'docs/v386_calibration_ablation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tasks=len(records),no_reselection=True)))


if __name__=='__main__':main()
