"""Read completed prediction caches; never fit missing models."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v108_supervised_transfer as v

if __name__=='__main__':
    d=v.load();records=[]
    for part in ('fold0','fold1'):
        for k,t in enumerate(d['tasks']):
            zs={kind:[v.cached(v.OUT/'models'/t/part/kind/str(seed)) for seed in v.SEEDS] for kind in ('transfer','scratch')}
            if any(z is None for a in zs.values() for z in a):continue
            q=zs['transfer'][0]['query'];y=d['y'][q];m=d['metrics'][k]
            ck=v.torch.load(v.OUT/'models'/t/part/'transfer'/str(v.SEEDS[0])/'model.pt',map_location='cpu',weights_only=False)
            state=ck['states'][k];ps={}
            for kind in zs:
                raw=np.mean([z['raw'] for z in zs[kind]],axis=0);p=raw*state['scale']+state['center']
                if m in ('AUROC','AUPRC'):p=expit(p)
                if t=='ppbr_az':p=p.clip(0,100)
                ps[kind]=p
            old=v.baseline(d,k,part);new=v.mixed(m,old,ps['transfer'])
            r=dict(task=t,part=part,metric=m,baseline=round(v.r83.score(m,y,old),5),transfer=round(v.r83.score(m,y,ps['transfer']),5),scratch=round(v.r83.score(m,y,ps['scratch']),5),gain=round(v.r83.gain(m,y,old,new),5),transfer_gain=round(v.r83.gain(m,y,ps['scratch'],ps['transfer']),5))
            records.append(r);v.emit(**r)
    v.data.save(v.OUT/'development_snapshot.json',dict(status='INTERIM_ONLY_NO_SELECTION',records=records))
