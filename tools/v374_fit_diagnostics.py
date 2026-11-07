"""Read fit-only replicate diagnostics; no estimator fitting or held labels."""
import json
import numpy as np
import v374_specialists as m
def main():
    d=m.load();records=[]
    for t in m.TASKS:
        for part in ('fold0','fold1'):
            a=m.design(d,t,part);fit=a['fit'];w,au=m.replicate_weights(d,fit);yy=d['y'][fit]
            keys=d['canonical'][d['mi'][fit]];u,inv=np.unique(keys,return_inverse=True)
            mixed=[]
            for i in range(len(u)):
                ix=np.flatnonzero(inv==i)
                if len(ix)>1:mixed.append(dict(count=len(ix),range=float(np.ptp(yy[ix])),stereo_identities=len(np.unique(d['isomeric'][d['mi'][fit[ix]]]))))
            records.append(dict(task=t,part=part,fit_rows=len(fit),isomeric_audit=au,canonical_duplicate_groups=len(mixed),canonical_mixed_stereo=sum(r['stereo_identities']>1 for r in mixed),
                canonical_conflicting_duplicates=sum(r['range']>1e-8 for r in mixed),weight_min=float(w.min()),weight_max=float(w.max()),target_quantiles=np.quantile(yy,[0,.1,.5,.9,1]).tolist()))
    m.save(m.ROOT/'fit_only_diagnostics.json',dict(status='COMPLETE',fit_labels_only=True,model_selection_unchanged=True,records=records))
    print(json.dumps([{k:r[k] for k in ('task','part','fit_rows','canonical_duplicate_groups','canonical_conflicting_duplicates','canonical_mixed_stereo')} for r in records]))
if __name__=='__main__':main()
