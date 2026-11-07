"""Read existing development predictions only; never select on confirmation."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parent
def main():
    d=np.load(ROOT/'official_rows.npz');out=[]
    for path in sorted((ROOT/'outputs/tasks').glob('*/fold[01]/*/predictions.npz')):
        t,part,config=path.parts[-4:-1];c=np.load(ROOT/'controls'/t/(part+'.npz'));z=np.load(path);y=d['y'][c['query']];b=np.abs(y-c['p']).mean()
        out.append(dict(task=t,part=part,config=config,baseline=float(b),expert=float(np.abs(y-z['p']).mean()),gain_at_20percent=float(1-np.abs(y-(.8*c['p']+.2*z['p'])).mean()/b)))
    print(json.dumps(out))
if __name__=='__main__':main()
