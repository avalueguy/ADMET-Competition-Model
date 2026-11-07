"""Read two primary method sources only; no datasets or pretrained weights."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,re,urllib.request
from html import unescape
ROOT=Path(__file__).resolve().parents[1]
URLS=(('recursive_feature_machines','https://raw.githubusercontent.com/aradha/recursive_feature_machines/main/README.md'),('recursive_feature_paper','https://arxiv.org/abs/2212.13881'))
def fetch(item):
 key,url=item
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=25)as response:payload=response.read()
  raw=payload.decode('utf-8',errors='replace');clean=unescape(re.sub('<[^>]+>',' ',raw));clean=re.sub(r'\s+',' ',clean)
  return dict(key=key,url=url,status='READ',sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload),excerpt=clean[:8000])
 except Exception as exc:return dict(key=key,url=url,status='UNAVAILABLE',error=str(exc))
def main():
 with ThreadPoolExecutor(max_workers=2)as pool:records=list(pool.map(fetch,URLS))
 value=dict(scope='Two selected primary sources, not exhaustive worldwide review.',records=records,
  application='Task-specific diagonal metric learned from squared input gradients. Our native-median SVR, damping, dimensional truncation and held-group protocol are independent choices, not a claim of paper-equivalent implementation or benchmark accuracy.',
  no_dataset_or_weight_download=True,no_fitting=True)
 (ROOT/'docs/v384_primary_references.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(value,ensure_ascii=False))
if __name__=='__main__':main()
