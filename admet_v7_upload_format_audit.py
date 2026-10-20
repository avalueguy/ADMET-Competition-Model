"""Inspect CSV serialization and header positions without changing predictions."""
import csv
import io
import json
import hashlib
from pathlib import Path

paths=list(Path('admet_submissions/v7').glob('*.csv'))+list(Path('admet_submissions/v6').glob('*.csv'))
paths += [Path('admet_official/submission_template.csv')]
paths += list(Path('admet_submissions').glob('*无表头.csv'))
reports=[]
for p in paths:
    b=p.read_bytes();rows=list(csv.reader(io.StringIO(b.decode('utf-8-sig'))))
    reports.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),physical_lines=len(b.splitlines()),
        csv_rows=len(rows),first=rows[0][:3],last=rows[-1][:3],
        header_rows=[i+1 for i,r in enumerate(rows) if r and r[0].lstrip('\ufeff')=='task'],
        widths=sorted(set(map(len,rows))),bom=b.startswith(b'\xef\xbb\xbf'),crlf=b.count(b'\r\n'),
        bare_cr=b.count(b'\r')-b.count(b'\r\n'),tail=repr(b[-170:])))
print(json.dumps(reports,ensure_ascii=True,indent=2))
