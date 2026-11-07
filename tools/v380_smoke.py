"""Real remote convex fit and independent kernel sanity checks before launch."""
from pathlib import Path
import json
import joblib
import numpy as np
import v380_histogram_median as m


def main():
    m.setup()
    d=m.load()
    a=m.design(d,m.TASKS[0],'fold0')
    a={**a,'fit':a['fit'][:70],'query':a['query'][:9],'x':a['x'][:70],'qx':a['qx'][:9]}
    k,q=m.matrices(d,a)
    values=[]
    for name in k:
        assert np.linalg.eigvalsh(k[name].astype(float)).min()>-1e-4
        model=m.SVR(kernel='precomputed',C=3,epsilon=0,tol=1e-5,max_iter=1000000)
        y=d['y'][a['fit']]
        model.fit(k[name],y)
        assert model.fit_status_==0
        path=m.ROOT/('smoke_'+name+'.joblib')
        joblib.dump(model,path)
        np.testing.assert_array_equal(model.predict(q[name]),joblib.load(path).predict(q[name]))
        # Block reordering must only reorder the kernel rows/columns.
        raw=d['fingerprints'][d['mi'][a['fit']],:2048]
        np.testing.assert_array_equal(m.histogram_map(raw)[::-1],m.histogram_map(raw[::-1]))
        values.append(dict(kernel=name,converged=True,iterations=int(model.n_iter_),replay_error=0.))
    result=dict(status='PASS',signature=m.signature(),records=values,all_fitting_remote=True,outer_scores_computed=False)
    (m.ROOT/'smoke_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
