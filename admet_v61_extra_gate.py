"""Independent new scaffold split for chemical-only winners, no supervised graph features."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import json
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
import admet_v24_models as b
import admet_v61_run as v

def main():
    b.remote_only();d=b.build_data();views=v.chemistry(d);report=json.loads((v.OUT/'fresh.json').read_text());records=[]
    for r in report['tasks']:
        if not r['pass_gate'] or v.P['configurations'][r['selection']['variant']][0]!=0:continue
        task=r['task'];k=d['tasks'].index(task);m=r['metric'];s=r['selection'];allrows=np.flatnonzero((d['tid']==k)&(d['split']<2));groups=np.unique(d['rg'][allrows]);seed=v.P['fresh_seed']+200+k;held=np.random.default_rng(seed).permutation(groups)[:max(1,int(.25*len(groups)))];active=np.isin(d['rg'][allrows],held);rr,q=allrows[~active],allrows[active]
        pp=v.fit(d,None,views,k,rr,q,'fresh_independent',v.OUT/task/'fresh_independent',[s['variant']])[s['variant']];raw=np.concatenate([d['fp2'],d['fcfp'],d['desc'],d['emb']],1);pre=b.Preprocessor().fit(raw[np.unique(d['mi'][rr])]);xx=pre.transform(raw);cls=m in ['AUROC','AUPRC'];tf=b.Target(d['y'][rr],m);model=(ExtraTreesClassifier if cls else ExtraTreesRegressor)(n_estimators=350,min_samples_leaf=2,max_features=.6,n_jobs=2,random_state=seed);model.fit(xx[d['mi'][rr]],tf.encode(d['y'][rr]));ctrl=model.predict_proba(xx[d['mi'][q]])[:,1] if cls else model.predict(xx[d['mi'][q]])
        if m=='MAE':ctrl=tf.decode(ctrl)
        merged=b.blend(m,ctrl,pp,s['weight']);gain=b.delta(m,d['y'][q],ctrl,merged);records.append(dict(task=task,gain=gain,rows=len(q),pass_gate=bool(gain>0),scaffold_overlap=0,control='ExtraTrees freshly retrained under identical new split; not historical ensemble'));b.emit('v61_independent',**records[-1])
    b.save(v.OUT/'independent_chemical_gate.json',dict(status='COMPLETE',signature=v.signature(),tasks=records))

if __name__=='__main__':main()
