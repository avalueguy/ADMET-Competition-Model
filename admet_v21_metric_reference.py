"""Read public evaluator source only; never downloads datasets or assay labels."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_diagnostics/v21_postmortem/public_sources'
OUT.mkdir(parents=True,exist_ok=True)
urls={
    'tdc_evaluator.py':'https://raw.githubusercontent.com/mims-harvard/TDC/main/tdc/evaluator.py',
    'frontend_current_894.js':'http://www.aiinnovation.com.cn/web/static/js/894.2c9c8bac.js',
}
records=[]
for name,url in urls.items():
    try:
        with urllib.request.urlopen(url,timeout=15) as response:
            blob=response.read()
        if name.endswith('.js') and blob.lstrip().lower().startswith(b'<!doctype'):
            raise ValueError('HTML fallback returned, not JavaScript')
        (OUT/name).write_bytes(blob)
        records.append(dict(name=name,url=url,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)))
        text=blob.decode('utf-8',errors='replace')
        if 'tdc' in name:
            lines=text.splitlines()
            for i,line in enumerate(lines):
                if any(word in line.lower() for word in ['pr_auc','pr-auc','average_precision','precision_recall','from .utils']):
                    print(name,i+1,'\n'.join(lines[max(0,i-2):i+5]))
    except Exception as exc:
        records.append(dict(name=name,url=url,error=str(exc)))
(OUT/'provenance.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records),flush=True)
