"""Report effective scaffold counts without changing any selected models."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import admet_v77_ranking as data

def main():
    d=data.load_data();report=json.loads((ROOT/'admet_model_outputs_v144/independent_verification.json').read_text());records=[]
    for r in report['checks']:
        if 'scaffold_bootstrap_gain_ci95' not in r:continue
        q=np.load(ROOT/f"admet_model_outputs_v131/controls/{r['task']}/{r['part']}.npz")['query']
        _,counts=np.unique(d['groups'][q],return_counts=True)
        records.append(dict(task=r['task'],part=r['part'],query_rows=len(q),unique_scaffold_groups=len(counts),largest_group_rows=int(counts.max()),
            bootstrap_interpretable=bool(len(counts)>1),note='Single group: scaffold resampling cannot quantify uncertainty.' if len(counts)==1 else 'Cluster bootstrap diagnostic only; label reuse and selection bias remain.'))
    result=dict(status='DIAGNOSTIC_ONLY',records=records,selection_changed=False)
    path=ROOT/'docs/v144_scaffold_group_diagnostics.json';path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
