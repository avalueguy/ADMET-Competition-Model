"""Three isolated task shards; model recipes and signature remain unchanged."""
import argparse,json,fcntl
import numpy as np
import torch
import admet_v24_models as b
import admet_v33_run as run

def serialize_boosting():
    # Multiple simultaneous CatBoost GPU fits have no throughput benefit on one GPU.
    # Only scheduling changes; parameters, fitting rows and predictions are unchanged.
    for klass in [run.CatBoostClassifier,run.CatBoostRegressor]:
        original=klass.fit
        def fit_locked(self,*args,_original=original,**kwargs):
            with (run.OUT/'gpu_boost.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                return _original(self,*args,**kwargs)
        klass.fit=fit_locked

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['evaluate','final']);p.add_argument('shard',type=int);args=p.parse_args()
    b.remote_only();torch.backends.cuda.matmul.allow_tf32=True
    serialize_boosting()
    run.OUT.mkdir(parents=True,exist_ok=True);all_tasks=run.P['tasks'].copy();tasks=all_tasks[args.shard::3]
    d=b.build_data();sig=run.signature()
    if args.stage=='evaluate':
        report=run.OUT/f'analysis_shard{args.shard}.json'
        if report.exists():
            old=json.loads(report.read_text())
            if old['status']=='EVALUATED' and old['signature']==sig:raise SystemExit(0)
        save=b.save
        def shard_save(path,obj):
            path=run.OUT/f'analysis_shard{args.shard}.json' if path==run.OUT/'analysis.json' else path
            save(path,obj)
        b.save=shard_save;run.P['tasks']=tasks
        run.evaluate(d,run.other_cache(d,[0]))
    else:
        report=json.loads((run.OUT/'analysis.json').read_text());assert report['signature']==sig
        selected=[r for r in report['tasks'] if r['selection'] and r['task'] in tasks]
        compact={}
        if selected:
            other=run.other_cache(d,[0,1])
            for r in selected:
                k=d['tasks'].index(r['task']);rows=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2))
                compact[r['task']]=run.fit(d,k,rows,q,[0,1],other,run.OUT/r['task']/'final').tolist()
        b.save(run.OUT/f'test_predictions_shard{args.shard}.json',dict(signature=sig,predictions=compact))
    b.emit('v33_worker_completed',shard=args.shard,stage_requested=args.stage)
