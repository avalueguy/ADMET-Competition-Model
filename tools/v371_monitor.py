"""Read-only compact remote progress."""
from pathlib import Path
import json,time
ROOT=Path(__file__).resolve().parent


def main():
 p=ROOT/'run_status.json';status=json.loads(p.read_text()) if p.exists() else {}
 log=(ROOT/'training.log').read_text() if (ROOT/'training.log').exists() else ''
 rows=[]
 for line in log.splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 predictions=list((ROOT/'outputs/tasks').glob('*/*/*/predictions.npz'))
 value=dict(status=status,completed_jobs=len(predictions),last_records=rows[-2:],traceback_seen='Traceback (most recent call last)' in log)
 for n in ('development','confirmation'):
  p=ROOT/'outputs'/(n+'.json')
  if p.exists():
   a=json.loads(p.read_text());value[n]=[dict(task=r['task'],config=r.get('config',r.get('best',{}).get('config')),weight=r.get('weight',r.get('best',{}).get('weight')),gains={v['part']:v['gain'] for v in r.get('confirmation',r.get('best',{}).get('parts',[]))},strict=r.get('strict_accepted'),experimental=r.get('experimental_eligible')) for r in a['records']]
 print(json.dumps(value))


if __name__=='__main__':main()
