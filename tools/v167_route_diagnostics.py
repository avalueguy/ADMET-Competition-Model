"""Describe frozen route outcomes without changing any selection decisions."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    output=[]
    for n in (149,150,151,153,157,159):
        folder=ROOT/f'admet_model_outputs_v{n}';cp=folder/'confirmation.json'
        if not cp.exists():continue
        c=json.loads(cp.read_text());records=[]
        for r in c.get('records',[]):
            gains={p['part']:round(p['gain'],6) for p in r['confirmation']}
            records.append(dict(task=r['task'],metric=r['metric'],config=r['config'],weight=r['weight'],gains=gains,
                reasons=(['nonpositive_point_gain'] if min(gains.values())<=0 else [])+['uncertain_'+b['part'] for b in r['bootstrap'] if not b['lower_positive']],bootstrap=r['bootstrap'],accepted=r['accepted']))
        output.append(dict(route=n,status=c['status'],records=records))
    out=ROOT/'admet_model_outputs_v161';out.mkdir(exist_ok=True)
    (out/'route_diagnostics.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps([{**r,'records':[{k:v for k,v in q.items() if k!='bootstrap'} for q in r['records']]} for r in output],indent=2),flush=True)
if __name__=='__main__':main()
