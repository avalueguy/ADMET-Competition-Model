"""Compare completed kernel configurations descriptively; no reselection/fits."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    dev=json.loads((ROOT/'experiments/regression9_v380/outputs/development.json').read_text())
    records=[]
    for r in dev['records']:
        details=[]
        for view in ('sqrt_polynomial','hellinger_gaussian'):
            choices=[x for x in r['choices'] if x['config']['view']==view]
            best=max(choices,key=lambda x:(x['objective'],-x['weight']))
            details.append(dict(view=view,config=best['config'],weight=best['weight'],
                                gains=[p['gain'] for p in best['parts']],
                                rank_gains=[p.get('rank_gain') for p in best['parts']],objective=best['objective']))
        records.append(dict(task=r['task'],frozen_best=r['best'],geometry_comparison=details))
    value=dict(signature=dev['signature'],development_fits=104,
               analysis='Descriptive geometry comparison, not confirmation retuning.',records=records)
    (ROOT/'docs/v380_geometry_comparison.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',tasks=len(records),development_fits=104,no_fitting=True)))


if __name__=='__main__':main()
