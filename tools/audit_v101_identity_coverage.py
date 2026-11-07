"""Structure-only coverage and train-only repeat-label consistency; no query labels."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v96_normalized_assays as v

def main():
    d=v.load();ids=np.load(v.IDENTITIES)['keys'];records=[]
    identities={'canonical':d['canonical'],'largest_parent':ids[d['mi'],0],'neutral_parent':ids[d['mi'],1],'tautomer':ids[d['mi'],2]}
    for k,task in enumerate(d['tasks']):
        train=np.flatnonzero((d['tid']==k)&(d['split']==0));valid=np.flatnonzero((d['tid']==k)&(d['split']==1));test=np.flatnonzero((d['tid']==k)&(d['split']==2))
        for name,key in identities.items():
            values={}
            for i in train:values.setdefault(str(key[i]),[]).append(float(d['y'][i]))
            repeat=[a for a in values.values() if len(a)>1];known=set(values);final_known=set(key[np.r_[train,valid]])
            rec=dict(task=task,identity=name,train_rows=len(train),train_unique=len(known),repeated_groups=len(repeat),
                train_repeat_nonzero_range=int(sum(np.ptp(x)>1e-8 for x in repeat)),
                valid_rows=len(valid),valid_covered=int(sum(str(key[i]) in known for i in valid)),
                test_rows=len(test),test_covered_by_train=int(sum(str(key[i]) in known for i in test)),
                test_covered_by_train_valid=int(sum(str(key[i]) in final_known for i in test)))
            records.append(rec)
    report=dict(status='COMPLETE',records=records,query_labels_read=False,test_labels_used=False)
    v.b.save(ROOT/'admet_model_outputs_v101/identity_coverage.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
