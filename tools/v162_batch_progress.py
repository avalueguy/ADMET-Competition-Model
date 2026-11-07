"""Compact six-route status, preserving the frozen acceptance rules."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    result=[]
    for v in (149,150,151,153,157,159):
        out=ROOT/f'admet_model_outputs_v{v}';p=out/'development.json';c=out/'confirmation.json';f=out/'frozen_selection.json'
        d=json.loads(p.read_text()) if p.exists() else {};cc=json.loads(c.read_text()) if c.exists() else {};ff=json.loads(f.read_text()) if f.exists() else {}
        cached=len([p for p in (out/'models').glob('*/fold1/*') if (p/'predictions.npz').exists()])
        result.append(dict(route=v,development_tasks=len({r['task'] for r in d.get('records',[])}),pretrained_task_configs=cached,development=d.get('status'),confirmation=cc.get('status'),nominees=[r['task'] for r in ff.get('selected',[])],accepted=[r['task'] for r in cc.get('records',[]) if r['accepted']]))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
