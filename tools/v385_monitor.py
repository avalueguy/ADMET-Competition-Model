"""Read compact progress; never fit or select a candidate."""
from pathlib import Path
import json,time,subprocess
ROOT=Path('/root/autodl-tmp/regression9_v385')
def main():
 result=dict(at_unix=time.time(),total_development_fits=296)
 for n in('run_status','launch','smoke_receipt'):
  p=ROOT/(n+'.json')
  if p.exists():result[n]=json.loads(p.read_text())
 paths=list((ROOT/'outputs/tasks').glob('*/*/*/predictions.npz'))
 result['completed_development_fits']=sum(p.parts[-3]in('fold0','fold1')for p in paths)
 result['prediction_packages']=len(paths)
 if(ROOT/'training.log').exists():
  events=[]
  for line in(ROOT/'training.log').read_text(errors='replace').splitlines():
   try:events.append(json.loads(line))
   except json.JSONDecodeError:pass
  if events:result['latest_event']={k:v for k,v in events[-1].items()if k in('stage','task','part','completed','total','config','seconds')}
 result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
 (ROOT/'progress.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
