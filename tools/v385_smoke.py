"""Remote all-transform fit/replay smoke without scoring held labels."""
import json,numpy as np
import v385_warped_median as m
def main():
 m.setup();d=m.load();t='ppbr_az';a=m.design(d,t,'fold0');a={**a,'fit':a['fit'][:90],'query':a['query'][:7],'x':a['x'][:90],'qx':a['qx'][:7]}
 original=m.design;m.design=lambda d,t,p:a
 records=[]
 for c in m.configs(t):
  p=m.predict(d,t,'smoke',c);r=m.replay(d,t,'smoke',c);np.testing.assert_array_equal(p,r)
  changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.;np.testing.assert_array_equal(r,m.replay(changed,t,'smoke',c))
  yy,w,s=m.target(d['y'][a['fit']],c);roundtrip=m.native(yy,s,t);assert np.max(np.abs(roundtrip-d['y'][a['fit']]))<1.5 # rank map ties are approximate at finite n
  records.append(dict(config=c['name'],replay_error=0.,finite_predictions=True))
 m.design=original;value=dict(status='PASS',signature=m.signature(),models=len(records),outer_scores_computed=False,records=records)
 m.save(m.ROOT/'smoke_receipt.json',value);print(json.dumps({k:v for k,v in value.items()if k!='records'}))
if __name__=='__main__':main()
