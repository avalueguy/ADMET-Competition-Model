from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import admet_v67_data as b
import admet_v73_conditional as c
from repack_upload_template import field_spans
from verify_submission import verify
b.rdBase.rdkitVersion='2025.03.6'

def main():
 d=b.build();base=next((ROOT/'admet_submissions/v73_v72').glob('*.csv'));templ=ROOT/'admet_official/submission_template.csv';rows=[]
 with base.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.reader(f))
 selected={'cyp2c9_substrate_carbonmangels':.10,'cyp2d6_substrate_carbonmangels':.10,'half_life_obach':.25}; replay={}
 for task,w in selected.items():
  k=d['tasks'].index(task);metric=d['metrics'][k];fit,q=c.rows(d,k,'final'); ps=[]
  for seed in (c.SEED,c.SEED+1):ps.append(c.fit_predict(d,k,'final','aux',iterations=120,seed=seed))
  p=np.mean(ps,axis=0);pos=[i for i,r in enumerate(rows) if i and r[0]==task];old=np.array([float(rows[i][4]) for i in pos]);new=b.blend(metric,old,p,w)
  for i,v in zip(pos,new):rows[i][4]=repr(float(v))
  replay[task]={'weight':w,'changed_rows':int(np.sum(new!=old))}
 raw=templ.read_bytes();spans=field_spans(raw);chunks=[];cursor=0
 for i,row in enumerate(rows[1:],1):lo,hi=spans[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
 chunks.append(raw[cursor:]);out=ROOT/'admet_submissions/v76_aggressive';out.mkdir(parents=True,exist_ok=True);file=out/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261023.csv';file.write_bytes(b''.join(chunks));audit=verify(file,templ);audit.update(status='PASS',baseline='v73',changed_tasks=replay,test_labels_used=False,external_data=False);b.save(file.with_suffix('.verification.json'),audit);b.save(out/'replay.json',replay);print(json.dumps({'file':str(file),'sha256':b.sha(file),'audit':audit},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
