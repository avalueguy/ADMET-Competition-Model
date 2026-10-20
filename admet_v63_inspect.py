import sys,inspect,json,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path('admet_runtime/v63_packages').resolve()))
from tabicl import TabICLClassifier,TabICLRegressor
print(inspect.signature(TabICLClassifier));print(inspect.signature(TabICLRegressor));print(inspect.getsource(TabICLRegressor.__init__))
for name in ['tabicl/sklearn/base.py','tabicl/sklearn/regressor.py','tabicl/sklearn/classifier.py']:
    file=Path('admet_runtime/v63_packages')/name
    if file.exists():
        lines=file.read_text().splitlines();print(json.dumps(dict(file=name,lines=[x for x in lines if any(s in x for s in ['checkpoint','model_path','repo_id','ckpt','download'])][:80])))
for host in ['https://hf-mirror.com/api/models/jingang/TabICL-clf','https://hf-mirror.com/api/models/jingang/TabICL-reg']:
    try:
        with urllib.request.urlopen(host,timeout=12) as r:d=json.load(r)
        print(json.dumps(dict(url=host,id=d.get('id'),siblings=d.get('siblings'),card=d.get('cardData'))))
    except Exception as e:print(json.dumps(dict(url=host,error=str(e))))
