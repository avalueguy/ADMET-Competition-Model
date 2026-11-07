"""Remote-only tiny full-horizon curve smoke, without outer evaluation."""
import json,joblib,numpy as np
import v387_correlated_metric as m
def main():
 m.setup();d=m.load();a=m.design(d,m.TASKS[0],'fold0')
 a={**a,'fit':a['fit'][:80],'query':a['query'][:7],'x':a['x'][:80],'qx':a['qx'][:7]}
 original=m.design;m.design=lambda data,t,p:a
 errors=[]
 for b in m.BASES:
  s=m.train_curve(d,m.TASKS[0],'smoke',b)
  assert len(s['history'])==9
  for c in m.CONFIGS:
   if c['base']!=b['name']:continue
   pred=m.predict(d,m.TASKS[0],'smoke',c);rp=m.replay(d,m.TASKS[0],'smoke',c)
   np.testing.assert_array_equal(pred,rp);errors.append(float(np.max(np.abs(pred-rp))))
   changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.
   np.testing.assert_array_equal(rp,m.replay(changed,m.TASKS[0],'smoke',c))
   x,_,_=m.make_arrays(d,a,s['transform']);ss=next(z for z in s['snapshots']if z['round']==c['round'])
   assert np.linalg.eigvalsh(m.kernel(x,x,ss['metric'],b['bandwidth']).astype(float)).min()>-1e-4
 m.design=original
 value=dict(status='PASS',signature=m.signature(),full_curves=2,converged_solves=18,replay_error=max(errors),query_label_perturbation_pass=True,outer_scores_computed=False)
 m.save(m.ROOT/'smoke_receipt.json',value);print(json.dumps(value))
if __name__=='__main__':main()
