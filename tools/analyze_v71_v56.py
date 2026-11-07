"""Audit completed specialist caches against explicitly identified history."""
from pathlib import Path
import sys, csv, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import admet_v67_data as b
import admet_v71_v56_catboost as v

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.reader(f))

def main():
    d=b.build()
    p56=next((ROOT/'admet_submissions/v56').glob('*.csv'))
    p57=next((ROOT/'admet_submissions/v57').glob('*.csv'))
    r56,r57=read_csv(p56),read_csv(p57)
    proxy_path=ROOT/'recovered_history/baseline_valid.npy'
    cp_path=ROOT/'recovered_history/baseline_cv_proxy.npy'
    base=np.load(proxy_path); cp=np.load(cp_path)
    assert base.shape==cp.shape==d['y'].shape
    old_scores={x['task']:x['base_valid'] for x in json.loads((ROOT/'admet_model_outputs_v49/fusion.json').read_text())['tasks']}
    folds=np.load(b.OUT/'folds.npy')
    records=[]
    for k,t in enumerate(d['tasks']):
        old=[r[4] for r in r56[1:] if r[0]==t]
        later=[r[4] for r in r57[1:] if r[0]==t]
        same=old==later
        vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        dest=v.OUT/t/'valid/predictions.npz'
        rec=dict(task=t,metric=d['metrics'][k],same_v56_v57=same,
                 historical_v56_valid_reference=old_scores.get(t),
                 proxy_identity='v57 development proxy',complete=False)
        if dest.exists():
            z=np.load(dest);np.testing.assert_array_equal(z['query'],vi)
            p=z['p'];m=d['metrics'][k]
            rec.update(complete=True,proxy_valid_score=b.metric_score(m,d['y'][vi],base[vi]),
                       specialist_valid_score=b.metric_score(m,d['y'][vi],p),weights=[])
            for w in (.1,.25,.5,.75,1.):
                new=v.blend(m,base[vi],p,w)
                gs=[]
                for f in range(3):
                    ix=np.flatnonzero((d['tid']==k)&(d['split']==0)&(folds==f))
                    zc=np.load(v.OUT/t/f'fold{f}/predictions.npz');np.testing.assert_array_equal(zc['query'],ix)
                    gs.append(b.gain(m,d['y'][ix],cp[ix],v.blend(m,cp[ix],zc['p'],w)))
                rec['weights'].append(dict(weight=w,valid_score=b.metric_score(m,d['y'][vi],new),
                    valid_gain=b.gain(m,d['y'][vi],base[vi],new),historical_proxy_cv_gains=gs,
                    historical_proxy_cv_mean=float(np.mean(gs))))
        records.append(rec)
    result=dict(status='DIAGNOSTIC',baseline_csv_sha256=b.sha(p56),
        v57_csv_sha256=b.sha(p57),proxy_valid_sha256=b.sha(proxy_path),
        proxy_cv_sha256=b.sha(cp_path),test_labels_used=False,tasks=records,
        limitations='Historical ensemble predictions are reused proxies on old folds. New folds differ. Only same_v56_v57 endpoints have identity support for the v56 proxy; this is not independent validation or a platform score.')
    b.save(ROOT/'admet_model_outputs_v72_v56/diagnostic.json',result)
    for x in records:
        if x['complete']:
            print(json.dumps(x,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
