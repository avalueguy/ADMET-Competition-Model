import pandas as pd, numpy as np, csv, io, hashlib, json
from pathlib import Path
root=Path('.')
base=next((root/'admet_submissions/v20').glob('*.csv'))
template=root/'admet_official/submission_template.csv'
out=root/'admet_submissions/v22'
out.mkdir(exist_ok=True)
rows=list(csv.reader(io.StringIO(base.read_bytes().decode('utf-8-sig'),newline='')))
# Only changes with direct, reproducible v21 test predictions and positive fixed validation evidence.
# v21 CYP2C9 validation was positive on OOF/official valid; PPBR was positive in prior v21 gate but current platform unchanged, keep unchanged pending evidence.
changes={}
for task in ['cyp2c9_substrate_carbonmangels']:
 p=root/'remote_extract_v22/admet_release_v21'/task/f'submission_{task}.csv'
 if p.exists(): changes[task]=pd.read_csv(p).Y.to_numpy(float)
# Conservative optional v16 routes: require current v16 validation gain > 0 and actual changed route.
# Do not apply because historical v16 was not platform-tested after v20 anchor.
for row in rows[1:]:
 t=row[0]
 if t in changes:
  q=changes[t][int(row[1])]
  row[4]=repr(float(np.clip(q,0,1)))
trows=list(csv.reader(io.StringIO(template.read_bytes().decode('utf-8-sig'),newline='')))
assert len(rows)==len(trows)==16380 and rows[0]==trows[0]
for a,b in zip(rows[1:],trows[1:]): assert a[:4]+a[5:]==b[:4]+b[5:]
buf=io.StringIO(newline=''); csv.writer(buf,lineterminator='\r\n').writerows(rows)
path=out/'我从山北来带着兰花草-小分子药物预测技术创新赛-20260927-v22.csv'; path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode())
checks={'status':'VERIFIED_REMOTE_EVIDENCE_CANDIDATE','baseline':'v20','baseline_score':50.019727,'changed':sorted(changes),'rows':16379,'columns':6,'tasks':22,'encoding':'UTF-8 BOM','line_endings':'CRLF','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'platform_score':None,'note':'One-task CYP2C9 substrate replacement from remote v21 fixed validation candidate; no score guarantee.'}
(out/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
