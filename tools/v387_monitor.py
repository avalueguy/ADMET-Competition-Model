"""Read complete-curve progress only."""
from pathlib import Path
import json,time,subprocess
ROOT=Path('/root/autodl-tmp/regression9_v387')
def main():
 x=dict(at_unix=time.time(),total_development_solves=324)
 for n in('run_status','launch','smoke_receipt'):
  p=ROOT/(n+'.json')
  if p.exists():x[n]=json.loads(p.read_text())
 x['completed_development_solves']=9*sum(p.parts[-3]in('fold0','fold1')for p in(ROOT/'outputs/curves').glob('*/*/*/state.joblib'))
 x['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
 (ROOT/'progress.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items()if k not in('launch','smoke_receipt')}))
if __name__=='__main__':main()
