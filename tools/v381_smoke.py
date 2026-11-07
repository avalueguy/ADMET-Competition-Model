"""Remote-only real base fits, grouped OOF certificates, reload smoke."""
import json
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
import v381_oof_pareto as m


def main():
    m.setup();d=m.load();a=m.design(d,m.TASKS[0],'fold0')
    a={**a,'fit':a['fit'][:90],'query':a['query'][:9],'x':a['x'][:90],'qx':a['qx'][:9]}
    t,p,tag=m.TASKS[0],'smoke','full'
    s=m.fit_base(d,a,t,p,tag);rp=m.replay_base(d,a,t,p,tag)
    np.testing.assert_array_equal(rp,s['predictions'])
    k,q=m.matrices(d,a)
    for v in k:assert np.linalg.eigvalsh(k[v].astype(float)).min()>-1e-4
    oof=np.full((90,3),np.nan)
    for i,(tr,ho) in enumerate(GroupKFold(3).split(a['fit'],groups=d['rg'][a['fit']])):
        sub=m.subdesign(a,tr,ho)
        oof[ho]=m.fit_base(d,sub,t,p,'inner'+str(i))['predictions']
    certificates=[]
    for c in m.CONFIGS:
        stack=m.fit_simplex(oof,d['y'][a['fit']],d['rg'][a['fit']],c)
        cert=m.check_certificate(oof,d['y'][a['fit']],d['rg'][a['fit']],c,stack)
        path=m.ROOT/('smoke_'+c['name']+'.joblib');joblib.dump(stack,path)
        z=joblib.load(path);np.testing.assert_array_equal(z['weights'],stack['weights'])
        certificates.append(dict(config=c['name'],certificate=cert,weights=stack['weights'].tolist()))
    out=dict(status='PASS',signature=m.signature(),base_models=12,base_replay_error=0.,
             all_fitting_remote=True,outer_scores_computed=False,certificates=certificates)
    m.save(m.ROOT/'smoke_receipt.json',out);print(json.dumps(out))


if __name__=='__main__':main()
