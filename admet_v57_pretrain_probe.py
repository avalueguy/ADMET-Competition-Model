"""Read public pretrained-model metadata; no competition labels or predictions involved."""
import json
import os
import urllib.request
assert os.name!='nt'
for name in ['DeepChem/ChemBERTa-77M-MTR','DeepChem/ChemBERTa-10M-MTR']:
    try:
        with urllib.request.urlopen('https://huggingface.co/api/models/'+name,timeout=20) as response:a=json.load(response)
        print(json.dumps(dict(model=name,sha=a.get('sha'),siblings=[x['rfilename'] for x in a.get('siblings',[])],card=a.get('cardData',{}))),flush=True)
    except Exception as e:print(name,type(e).__name__,str(e),flush=True)
