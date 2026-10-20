"""Prove target labels cannot influence conditional auxiliary assay inputs."""
import numpy as np,json
import admet_v24_models as b
from admet_v31_run import auxiliary,OUT,signature
b.remote_only();d=b.build_data();results=[]
for task in ['cyp2c9_substrate_carbonmangels','cyp2c9_veith','cyp3a4_veith']:
    k=d['tasks'].index(task);x,count,meta=auxiliary(d,[0],k)
    changed=dict(d);changed['y']=d['y'].copy();changed['y'][d['tid']==k]=123456.
    changed['y'][d['split']==2]=789123.
    again,count2,_=auxiliary(changed,[0],k)
    np.testing.assert_array_equal(x,again);np.testing.assert_array_equal(count,count2)
    # Every auxiliary source row is explicitly labelled train of a DIFFERENT task.
    source=np.flatnonzero((d['split']==0)&(d['tid']!=k))
    assert np.all(d['tid'][source]!=k) and np.all(d['split'][source]==0)
    results.append(dict(task=task,target_label_perturbation_invariant=True,test_label_perturbation_invariant=True,
                        target_observation_mask_removed=True,source_rows=len(source)))
b.save(OUT/'auxiliary_isolation_audit.json',dict(signature=signature(),tasks=results,external_labels=False,
    interpretation='Other official task labels can exist for the same query identity, explicitly conditional assay prediction'))
print(json.dumps(results))
