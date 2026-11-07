"""Read local method evidence and primary research pages; no model fitting."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import ast
import hashlib
import json
import re
import ssl
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
SOURCES=(
 ('D-MPNN molecular representations','https://arxiv.org/abs/1904.01561'),
 ('GROVER molecular pretraining','https://arxiv.org/abs/2007.02835'),
 ('MolCLR contrastive molecular learning','https://arxiv.org/abs/2102.10056'),
 ('TabR retrieval augmented tabular learning','https://arxiv.org/abs/2307.14338'),
 ('Tabular deep learning revisited','https://arxiv.org/abs/2106.11959'),
 ('CatBoost loss functions','https://catboost.ai/docs/en/concepts/loss-functions-regression'),
 ('XGBoost parameter reference','https://xgboost.readthedocs.io/en/stable/parameter.html'),
)


def fetch(item):
 name,url=item
 try:
  request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
  context=ssl.create_default_context()
  if hasattr(ssl,'enum_certificates'):
   for cert,encoding,trust in ssl.enum_certificates('ROOT'):
    if encoding=='x509_asn':
     try:context.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(cert))
     except ssl.SSLError:pass
  with urllib.request.urlopen(request,timeout=25,context=context) as stream:raw=stream.read(1500000)
  text=raw.decode('utf-8',errors='replace');plain=' '.join(re.sub('<[^>]+>',' ',text).split())
  title=re.search(r'<title>(.*?)</title>',text,re.S)
  return dict(topic=name,url=url,status='READ',title=' '.join(title.group(1).split()) if title else None,
              sha256=hashlib.sha256(raw).hexdigest(),excerpt=plain[:18000])
 except Exception as e:return dict(topic=name,url=url,status='UNAVAILABLE',error=type(e).__name__+': '+str(e)[:180])


def main():
 entries=[]
 for path in sorted((ROOT/'tools').glob('v*.py')):
  match=re.match(r'v(\d+)',path.name)
  if not match or not 145<=int(match[1])<=375:continue
  source=path.read_text(encoding='utf-8-sig')
  try:tree=ast.parse(source);doc=ast.get_docstring(tree) or ''
  except SyntaxError:continue
  method=[]
  for node in tree.body:
   if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('METHOD','TASKS','CONFIGS') for t in node.targets):
    method.append(ast.get_source_segment(source,node)[:5500])
  if not method or not any(s in (doc+' '.join(method)).lower() for s in ('mae','median','quantile','regress','spearman','clearance')):continue
  entries.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),doc=doc,configuration=method))
 documents=[]
 for pattern in ('*RESULT*.md','*OPTIMIZATION*.md','*DECISIONS*.md','*DIAGNOSTICS*.md','V145_RESULTS_AND_CODE_GUIDE.md'):
  for p in (ROOT/'docs').glob(pattern):
   text=p.read_text(encoding='utf-8-sig')
   if any(s in text for s in ('MAE','中位数','回归')):documents.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=text[:20000]))
 with ThreadPoolExecutor(max_workers=4) as pool:external=list(pool.map(fetch,SOURCES))
 value=dict(at_utc=datetime.now(timezone.utc).isoformat(),scope='Available local regression method source declarations and selected local result documents; seven selected primary web pages. Not an exhaustive world literature search.',
            methods=entries,documents=list({r['path']:r for r in documents}.values()),research=external)
 dest=ROOT/'docs/v376_material_review.json';dest.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(local_methods=len(entries),local_result_documents=len(value['documents']),research=[{k:v for k,v in r.items() if k in ('topic','url','status','title','error')} for r in external])))


if __name__=='__main__':main()
