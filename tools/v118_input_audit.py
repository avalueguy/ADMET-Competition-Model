"""Verify target exclusion without changing frozen experimental source files."""
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v117_conditional_structure as v

def run():
    d=v.prev.load();records=[]
    assert not np.isfinite(d['y'][d['split']==2]).any()
    for task in v.TASKS:
        k=d['tasks'].index(task)
        for part in ('fold0','fold1','fold2','valid','final'):
            a=v.design(d,k,part,'full');fit,q,x,qx,source=a
            assert not np.any(d['tid'][source]==k)
            original=d['y'][q].copy();d['y'][q]=987654.
            d.pop('_v96_assays',None)
            z=v.design(d,k,part,'full')
            d['y'][q]=original;d.pop('_v96_assays',None)
            for before,after in zip(a,z):np.testing.assert_array_equal(before,after)
            assert np.isfinite(d['y'][fit]).all()
            records.append(dict(task=task,part=part,fit=len(fit),query=len(q),features=x.shape[1],source_rows=len(source)))
    report=dict(status='PASS',conditional_signature=v.sig(),checks=records,
        target_query_label_perturbation_invariance=True,target_assay_and_mask_excluded=True,
        test_labels_loaded=False,external_data_added=False)
    v.b.save(ROOT/'admet_model_outputs_v118/input_audit.json',report)
    print(json.dumps(dict(status='PASS',checks=len(records),test_labels_loaded=False)),flush=True)

if __name__=='__main__':run()
