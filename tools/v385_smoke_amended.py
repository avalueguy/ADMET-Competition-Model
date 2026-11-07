"""Correct the finite empirical-rank smoke assertion; frozen models unchanged.

Midrank-to-linear-quantile mapping is not an identity at finite n. Test range
and monotonicity there; use exact round-trip checks only for analytic warps.
"""
import json,numpy as np
import v385_warped_median as m
def main():
 m.setup();d=m.load();t='ppbr_az';a=m.design(d,t,'fold0');a={**a,'fit':a['fit'][:90],'query':a['query'][:7],'x':a['x'][:90],'qx':a['qx'][:7]}
 original=m.design;m.design=lambda d,t,p:a;records=[];y=d['y'][a['fit']]
 for c in m.configs(t):
  pred=m.predict(d,t,'smoke',c);rp=m.replay(d,t,'smoke',c);np.testing.assert_array_equal(pred,rp)
  changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.;np.testing.assert_array_equal(rp,m.replay(changed,t,'smoke',c))
  yy,w,s=m.target(y,c);roundtrip=m.native(yy,s,t)
  if c['mode']=='gaussian_rank':
   ix=np.argsort(y,kind='stable');assert np.diff(roundtrip[ix]).min()>-1e-10 and roundtrip.min()>=y.min()and roundtrip.max()<=y.max()
  else:np.testing.assert_allclose(roundtrip,y,rtol=1e-10,atol=1e-10)
  records.append(dict(config=c['name'],replay_error=0.,transform_check='monotone_bounded'if c['mode']=='gaussian_rank'else'exact_roundtrip'))
 m.design=original
 value=dict(status='PASS',signature=m.signature(),models=len(records),outer_scores_computed=False,records=records,
  amendment='Original fixed1.5 rank roundtrip tolerance was unjustified. Frozen model/feature/evaluation code unchanged; replaced smoke with monotonicity/range assertion before any development fit.',
  amendment_sha256=m.sha(__file__))
 m.save(m.ROOT/'smoke_receipt.json',value);print(json.dumps({k:v for k,v in value.items()if k!='records'}))
if __name__=='__main__':main()
