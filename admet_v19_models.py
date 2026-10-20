"""v19 model families: wider attention network and deeper embedding boosting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from admet_v16_models import neural_fit, boost_fit, views, ViewAttention, boost_x, decode
from admet_v6_models import Preprocess

PROTOCOL = Path("admet_v19_protocol.json")

def recipe():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))

def fit(name, d, ix, y, metric):
    assert name.startswith("v19:")
    p = recipe(); family = name.split(":", 1)[1]
    ix = np.asarray(ix, dtype=int); y = np.asarray(y, dtype=float)
    if family == "view_attention_v2":
        m = neural_fit(d, ix, y, metric, p)
    elif family == "mol_boost_v2":
        m = boost_fit(d, ix, y, metric, p)
    else:
        raise KeyError(family)
    m["family"] = "view_attention" if family == "view_attention_v2" else "mol_boost"
    m["feature_key"] = d["v13_feature_key"]
    return dict(name=name, model=m, protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), fit_rows=ix,
                fit_target_sha256=hashlib.sha256(np.asarray(y, dtype="<f8").tobytes()).hexdigest())

def predict(bundle, d, ix):
    m = bundle["model"]; assert m["feature_key"] == d["v13_feature_key"]
    ix = np.asarray(ix, dtype=int)
    if m["family"] == "mol_boost":
        x = m["pre"].transform(boost_x(d, ix))
        q = [a.predict_proba(x)[:, 1] if m["classification"] else m["target"].decode(a.predict(x)) for a in m["models"]]
        return np.mean(q, axis=0)
    raw = views(d, ix); x = [pp.transform(a) for pp, a in zip(m["pre"], raw)]
    net = ViewAttention(m["dimensions"], m["hidden"], m["view_dropout"]).cuda(); out = []
    for state in m["models"]:
        net.load_state_dict(state); net.eval(); chunks = []
        with torch.inference_mode():
            for start in range(0, len(ix), 256):
                chunks.append(net([torch.tensor(a[start:start+256], device="cuda") for a in x]).cpu().numpy())
        out.append(decode(np.concatenate(chunks), m["target"]))
    del net
    q = np.mean(out, axis=0)
    assert q.shape == (len(ix),) and np.isfinite(q).all()
    return q
