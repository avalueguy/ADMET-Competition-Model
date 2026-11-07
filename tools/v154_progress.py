"""Compact broad-training progress, without selecting from confirmation."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    records=[]
    for version in (149,150,151,153):
        out=ROOT/f'admet_model_outputs_v{version}';dev=out/'development.json';con=out/'confirmation.json';fr=out/'frozen_selection.json'
        d=json.loads(dev.read_text()) if dev.exists() else {};c=json.loads(con.read_text()) if con.exists() else {};f=json.loads(fr.read_text()) if fr.exists() else {}
        best={}
        for r in d.get('records',[]):
            if r['task'] not in best or r['objective']>best[r['task']]['objective']:best[r['task']]=r
        positive={t:round(r['objective'],5) for t,r in best.items() if r['objective']>0}
        records.append(dict(version=version,developed=len(best),development_status=d.get('status'),confirmation_status=c.get('status'),frozen=[r['task'] for r in f.get('selected',[])],accepted=c.get('accepted'),positive_development=positive))
    print(json.dumps(records,indent=2),flush=True)
if __name__=='__main__':main()
