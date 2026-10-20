import json,importlib.util,urllib.request
for name in ['tabicl','tabpfn','einops','huggingface_hub','sklearn']:
    spec=importlib.util.find_spec(name);print(json.dumps(dict(package=name,installed=bool(spec))),flush=True)
urls=['https://pypi.org/pypi/tabicl/json','https://huggingface.co/api/models/jingang/TabICL-clf','https://huggingface.co/api/models/jingang/TabICL-reg']
for url in urls:
    try:
        with urllib.request.urlopen(url,timeout=20) as r:data=json.load(r)
        if 'pypi' in url:data={k:data['info'].get(k) for k in ['name','version','requires_dist','home_page','project_urls']}
        else:data={k:data.get(k) for k in ['id','gated','siblings','cardData']}
        print(json.dumps(dict(url=url,data=data)),flush=True)
    except Exception as e:print(json.dumps(dict(url=url,error=str(e))),flush=True)
