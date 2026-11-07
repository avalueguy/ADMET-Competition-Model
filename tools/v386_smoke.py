"""Real grouped inner-OOF bases and calibration smoke on the remote only."""
import json,joblib,numpy as np
from sklearn.model_selection import GroupKFold
import v386_oof_calibrated as m
def main():
 m.setup();d=m.load();t=m.TASKS[0];a=m.design(d,t,'fold0');a={**a,'fit':a['fit'][:90],'query':a['query'][:9],'x':a['x'][:90],'qx':a['qx'][:9]}
 y=d['y'][a['fit']];oof=np.full((90,3),np.nan)
 for i,(tr,ho)in enumerate(GroupKFold(3).split(a['fit'],groups=d['rg'][a['fit']])):
  sub=m.subdesign(a,tr,ho);oof[ho]=m.fit_base(d,sub,t,'smoke','inner'+str(i))['predictions']
 full=m.fit_base(d,a,t,'smoke','full')['predictions'];np.testing.assert_array_equal(full,m.replay_base(d,a,t,'smoke','full'))
 certificates=[]
 for c in m.CONFIGS:
  if c['calibration']=='raw':continue
  j=m.BASES.index(c['base']);s=m.calibration.fit(oof[:,j],y,c['penalty']);cert=m.calibration.certificate(oof[:,j],y,s)
  sorted_query=np.linspace(oof[:,j].min()-5,oof[:,j].max()+5,100);assert np.diff(m.calibration.apply(sorted_query,s)).min()>0
  path=m.ROOT/('smoke_'+c['name']+'.joblib');joblib.dump(s,path);np.testing.assert_array_equal(m.calibration.apply(full[:,j],s),m.calibration.apply(full[:,j],joblib.load(path)))
  certificates.append(dict(config=c['name'],certificate=cert,knots=len(s['knots'])))
 value=dict(status='PASS',signature=m.signature(),base_models=12,calibrators=6,base_replay_error=0.,outer_scores_computed=False,certificates=certificates)
 m.save(m.ROOT/'smoke_receipt.json',value);print(json.dumps(value))
if __name__=='__main__':main()
