"""Report final expert shifts and exclusion-group coverage without retuning."""
import json,sys,csv,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v98_shape_experts as v

def rows(p):return list(csv.DictReader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def main():
    d=v.load();out=ROOT/'admet_model_outputs_v101';rel=json.loads((out/'release.json').read_text());old=rows(v.ANCHOR);new=rows(ROOT/rel['file']);records=[]
    for task,ch in rel['selected'].items():
        k=d['tasks'].index(task);q=v.r83.split(d,k,'final')[1];tr,valid=v.r83.split(d,k,'valid')
        a=np.array([float(r['y_pred']) for r in old if r['task']==task]);z=np.array([float(r['y_pred']) for r in new if r['task']==task])
        assert len(q)==len(z)
        def top_groups(ix):
            g,c=np.unique(d['groups'][ix],return_counts=True);return dict(unique=len(g),largest_fraction=float(c.max()/len(ix)))
        rec=dict(task=task,branch=ch['version'],metric=d['metrics'][k],rows=len(q),
            old_quantiles=np.quantile(a,[0,.01,.25,.5,.75,.99,1]).tolist(),new_quantiles=np.quantile(z,[0,.01,.25,.5,.75,.99,1]).tolist(),
            mean_absolute_shift=float(np.abs(z-a).mean()),max_absolute_shift=float(np.abs(z-a).max()),
            train_groups=top_groups(tr),valid_groups=top_groups(valid),test_groups=top_groups(q),
            final_conformer_coverage=float(d['conformer_features'][d['mi'][q],-1].mean()))
        records.append(rec)
    report=dict(status='COMPLETE',selection_effect='NONE_REPORT_ONLY',records=records,test_labels_used=False,
        limitations='Rank blends change probability scale without calibrated risk interpretation. Historical exclusion groups can be concentrated; validation is approximate and historically reused.')
    v.b.save(out/'release_diagnostics.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
