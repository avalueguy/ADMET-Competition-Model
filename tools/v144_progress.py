"""Compact snapshot of broad training batch and fixed development route leaders."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VERSIONS=(134,135,136,137,139,141,143)
def main():
    states=[];nominees={}
    for v in VERSIONS:
        folder=ROOT/f'admet_model_outputs_v{v}';p=folder/'development.json';c=folder/'confirmation.json'
        dev=json.loads(p.read_text()) if p.exists() else {};conf=json.loads(c.read_text()) if c.exists() else {}
        tasks=list(dict.fromkeys(r['task'] for r in dev.get('records',[])))
        states.append(dict(version=v,developed=len(tasks),development_status=dev.get('status'),confirmation_status=conf.get('status'),accepted=conf.get('accepted')))
        f=folder/'frozen_selection.json'
        if not f.exists():continue
        for r in json.loads(f.read_text())['selected']:
            if r['task'] not in nominees or r['objective']>nominees[r['task']]['objective']:
                match=next((z for z in conf.get('records',[]) if z['task']==r['task']),{})
                nominees[r['task']]=dict(version=v,objective=r['objective'],confirmed=match.get('accepted'))
    print(json.dumps(dict(states=states,provisional_development_winners=nominees),indent=2),flush=True)
if __name__=='__main__':main()
