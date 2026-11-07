"""Structure overlap, target scale and query distribution checks without tuning."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
OUT=ROOT/'admet_model_outputs_v163'
def main():
    d=v.load();records=[]
    for task in v.tasks(d):
        k=d['tasks'].index(task);parts=[]
        for part in ('fold0','fold1','fold2','valid','final'):
            fit,q=v.split(d,k,part);a=d['views']['physchem'][d['mi'][fit]].astype(float);z=d['views']['physchem'][d['mi'][q]].astype(float)
            med=np.nanmedian(a,axis=0);scale=np.nanquantile(a,.75,axis=0)-np.nanquantile(a,.25,axis=0);scale=np.where(np.isfinite(scale)&(scale>.01),scale,1);drift=np.abs((np.nanmedian(z,axis=0)-med)/scale)
            parts.append(dict(part=part,fit=len(fit),query=len(q),exact_identity_overlap=int(np.isin(d['canonical'][q],d['canonical'][fit]).sum()),neutral_identity_overlap=int(np.isin(d['neutral_identity'][q],d['neutral_identity'][fit]).sum()),train_label_quantiles=np.quantile(d['y'][fit],[0,.05,.5,.95,1]).tolist(),physchem_median_shift_quantiles=np.quantile(drift[np.isfinite(drift)],[.5,.9,.99]).tolist()))
        records.append(dict(task=task,parts=parts))
    OUT.mkdir(exist_ok=True);v.b.save(OUT/'audit.json',dict(status='COMPLETE',selection_effect='NONE',records=records,test_labels_used=False,external_data=False));print(json.dumps(records),flush=True)
if __name__=='__main__':main()
