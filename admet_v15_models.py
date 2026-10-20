"""Versioned adapter: reuse fully specified base learners, train only remotely."""
import hashlib,importlib,json
from pathlib import Path
import numpy as np


ALLOWED={6:['kernel_1','kernel_10','extra','xgb','dense','graph'],
    7:['pharm_kernel','pharm_cat','chemberta_ft'],8:['multiscale_kernel','gated_transfer'],
    9:['gated_short'],10:['graph_transfer'],11:['sparse_uniform','sparse_scaffold'],
    12:['boost_compact','boost_fragments','kernel_adaptive'],13:['mol_kernel','mol_fusion','mol_mlp']}


def recipe(v):return json.loads(Path(f'admet_v{v}_protocol.json').read_text())
def parse(name):
    v,f=name.split(':',1);v=int(v[1:]);assert f in ALLOWED[v];return v,f


def load_task(task):
    from admet_v13_models import load_task as load
    d=load(task)
    d['v12_feature_key']=hashlib.sha256(('v12|'+d['v11_feature_key']).encode()).hexdigest()
    return d


def initialize(name,d):
    v,f=parse(name)
    if v==6 and f.startswith('kernel') and 'kernel' not in d:
        from admet_v6_features import mixed_kernel
        d['kernel']=mixed_kernel(d)
    return v,f,importlib.import_module(f'admet_v{v}_models')


def fit(name,d,ix,y,metric):
    from admet_v6_runner import remote_only
    remote_only();v,f,m=initialize(name,d)
    bundle=m.fit(f,d,np.asarray(ix,int),np.asarray(y,float),metric,recipe(v))
    return dict(name=name,model=bundle,protocol_sha256=hashlib.sha256(Path(f'admet_v{v}_protocol.json').read_bytes()).hexdigest(),
        fit_rows=np.asarray(ix,int),fit_target_sha256=hashlib.sha256(np.asarray(y,dtype='<f8').tobytes()).hexdigest())


def predict(bundle,d,ix):
    v,f,m=initialize(bundle['name'],d)
    assert bundle['protocol_sha256']==hashlib.sha256(Path(f'admet_v{v}_protocol.json').read_bytes()).hexdigest()
    p=np.asarray(m.predict(bundle['model'],d,np.asarray(ix,int)),float)
    assert p.shape==(len(ix),) and np.isfinite(p).all()
    return p
