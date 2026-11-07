"""Summarize explicit development controls and official auxiliary coverage."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b

def main():
    d=b.build();known={}
    for i in np.flatnonzero(d['split']<2):
        known.setdefault(d['canonical'][d['mi'][i]],set()).add(int(d['tid'][i]))
    result=[]
    for k,t in enumerate(d['tasks']):
        rec=dict(task=t)
        for si,part in ((1,'valid'),(2,'test')):
            q=np.flatnonzero((d['tid']==k)&(d['split']==si))
            rec[part+'_n']=len(q)
            rec[part+'_official_other_assays']=sum(bool(known.get(d['canonical'][d['mi'][i]],set())-{k}) for i in q)
        result.append(rec);print(json.dumps(rec),flush=True)
    b.save(ROOT/'admet_model_outputs_v73/coverage.json',dict(tasks=result,test_labels_used=False))

if __name__=='__main__':main()
