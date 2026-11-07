"""Audit source identities before historical routing; invalid candidates never upload."""
import csv,hashlib,io,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'admet_submissions/v174_historical_feedback_blend'
BASE=next((ROOT/'admet_submissions/v145_feedback_repair').glob('*.csv'))
SOURCES={'caco2_wang':ROOT/'recovered_history/official_scorecard_sources/admet_submissions/v23/v23_upload.csv','clearance_microsome_az':next((ROOT/'recovered_history/official_scorecard_sources/admet_submissions/v46').glob('*.csv')),'herg':next((ROOT/'admet_submissions/v114_herg_history52').glob('*.csv'))}
def read(p):return list(csv.DictReader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True);base=read(BASE);records=[]
 for task,p in SOURCES.items():
  old=[r for r in base if r['task']==task];src=[r for r in read(p) if r['task']==task]
  assert len(old)==len(src)
  assert all({k:v for k,v in a.items() if k!='y_pred'}=={k:v for k,v in z.items() if k!='y_pred'} for a,z in zip(old,src))
  changed=sum(a['y_pred']!=z['y_pred'] for a,z in zip(old,src))
  records.append(dict(task=task,source_file=p.relative_to(ROOT).as_posix(),sha256=sha(p),changed_rows=changed,decision='IDENTICAL_TO_BASELINE' if not changed else 'REJECTED_OFFICIAL_TASK_REGRESSION',source_task_score=49.677228 if task=='herg' else None,baseline_task_score=49.832263 if task=='herg' else None))
 rejected=[]
 for p in OUT.glob('*.csv'):
  target=p.with_suffix('.csv.rejected');assert not target.exists()
  p.rename(target);rejected.append(dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target)))
 report=dict(status='REJECTED_NO_NEW_SUBMISSION',version='v174',baseline='v145',baseline_official_score=51.166,source_identity_audit=records,rejected_preliminary_files=rejected,submission_created=False,automatically_uploaded=False,official_score=None,withdrawn_estimated_score=51.188,reason='Two sources equal v145; v114 hERG is below v145. Historical mapping in the handoff summary was incorrect.',current_best_retained=True)
 (OUT/'no_candidate.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=True))
if __name__=='__main__':main()
