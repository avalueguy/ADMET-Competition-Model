"""Official-only sparse fragment kernel complement, trained concurrently with GPU v69."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
import argparse
import csv
import hashlib
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
from rdkit import rdBase
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetMorganFeatureAtomInvGen, GetTopologicalTorsionGenerator
from scipy import sparse
from sklearn.svm import SVC, SVR

import admet_v67_data as b
import admet_v69_aggressive as base

ROOT = b.ROOT
OUT = ROOT / "admet_model_outputs_v70"
P = json.loads((ROOT / "admet_v70_protocol.json").read_text(encoding="utf-8"))
WORKER_DATA = None
WORKER_VIEWS = None


def signature():
    return hashlib.sha256((base.signature()+b.sha(__file__)+b.sha(ROOT/"admet_v70_protocol.json")).encode()).hexdigest()


def chemistry(d):
    folder = OUT / "features"
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(recipe="sparse_log_count_views_v1", rdkit=rdBase.rdkitVersion,
        smiles_sha256=hashlib.sha256(json.dumps(d["smiles"]).encode()).hexdigest(), views=P["views"])
    path = folder / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == meta
        return [sparse.load_npz(folder/(name+".npz")) for name in P["views"]]
    generators = [GetMorganGenerator(radius=radius, includeChirality=True) for radius in (1,2,3)]
    generators += [GetMorganGenerator(radius=2, includeChirality=True,
        atomInvariantsGenerator=GetMorganFeatureAtomInvGen()), GetTopologicalTorsionGenerator(includeChirality=True)]
    vocab = [{} for _ in generators]
    columns = [[] for _ in generators]
    values = [[] for _ in generators]
    pointers = [[0] for _ in generators]
    for i, sm in enumerate(d["smiles"]):
        mol, _ = b.parse_molecule(sm)
        for j, generator in enumerate(generators):
            for fragment, count in sorted(generator.GetSparseCountFingerprint(mol).GetNonzeroElements().items()):
                column = vocab[j].setdefault(fragment, len(vocab[j]))
                columns[j].append(column)
                values[j].append(np.log1p(count))
            pointers[j].append(len(columns[j]))
        if i % 5000 == 0:
            b.emit("v70_features", done=i, total=len(d["smiles"]))
    result = []
    for j, name in enumerate(P["views"]):
        x = sparse.csr_matrix((np.array(values[j],dtype=np.float32),
            np.array(columns[j],dtype=np.int32), np.array(pointers[j],dtype=np.int64)),
            shape=(len(d["smiles"]),len(vocab[j])))
        x.sort_indices()
        sparse.save_npz(folder/(name+".npz"), x)
        result.append(x)
    b.save(path, meta)
    b.save(folder/"audit.json",dict(status="PASS",label_free_vocabulary=True,
        dimensions=[x.shape[1] for x in result], nonzeros=[x.nnz for x in result],
        official_molecules=len(d["smiles"]), no_fixed_size_folding=True,
        intrinsic_hash_collisions_possible=True))
    return result


def matrices(d, views, rows, query, family):
    ref, qry = d["mi"][rows], d["mi"][query]
    unique = np.unique(ref)
    train = np.zeros((len(ref), len(ref)), dtype=np.float32)
    test = np.zeros((len(qry), len(ref)), dtype=np.float32)
    for source, weight in zip(views, family["weights"]):
        if not weight:
            continue
        x = source.astype(np.float64)
        binary = source.astype(np.float64)
        binary.data.fill(1.)
        if not family["counts"]:
            x = binary.copy()
        if family["idf_power"]:
            df = np.asarray(binary[unique].sum(0)).ravel()
            idf = (1+np.log((1+len(unique))/(1+df)))**family["idf_power"]
            x = x.multiply(idf).tocsr()
        r, q = x[ref], x[qry]
        rn = np.asarray(r.multiply(r).sum(1)).ravel()
        qn = np.asarray(q.multiply(q).sum(1)).ravel()
        for left, norm, output in ((r, rn, train), (q, qn, test)):
            for start in range(0, left.shape[0], 256):
                dot = (left[start:start+256] @ r.T).toarray()
                tanimoto = dot/np.maximum(norm[start:start+256,None]+rn[None]-dot, 1e-8)
                output[start:start+len(dot)] += weight*(.7*tanimoto+.3*tanimoto*tanimoto)
    return ((train+train.T)*.5).astype(np.float64), test.astype(np.float64)


def fit(d, views, k, partition):
    rows, query = base.splits(d, k, partition)
    folder = OUT / d["tasks"][k] / partition
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "predictions.npz"
    sig = signature()
    if path.exists():
        metadata = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
        assert metadata["signature"] == sig
        for state in metadata["models"]:
            assert b.sha(folder/state["file"]) == state["sha256"]
        with np.load(path, allow_pickle=False) as z:
            assert str(z["signature"]) == sig
            np.testing.assert_array_equal(z["fit"], rows)
            np.testing.assert_array_equal(z["query"], query)
            return z["p"].copy()
    started = time.monotonic()
    metric = d["metrics"][k]
    target = b.target(d["y"][rows], metric)
    y = b.encode(d["y"][rows], target)
    _, inverse, counts = np.unique(d["canonical"][d["mi"][rows]],return_inverse=True,return_counts=True)
    weights = 1/counts[inverse]; weights /= weights.mean()
    result, models = [], []
    for family, spec in enumerate(P["kernels"]):
        train, test = matrices(d, views, rows, query, spec)
        for c in P["c_values"]:
            cls = metric in ("AUROC","AUPRC")
            net = SVC(C=c,kernel="precomputed",cache_size=512) if cls else SVR(
                C=c,kernel="precomputed",epsilon=P["regression_epsilon"],cache_size=512)
            net.fit(train,y,sample_weight=weights)
            def predict(model):
                p = model.decision_function(test) if cls else model.predict(test)
                return p*target["scale"]+target["center"] if metric == "MAE" else p
            p = predict(net)
            assert np.isfinite(p).all()
            file = folder/f"family{family}_C{c:g}.joblib"
            joblib.dump(dict(model=net,target=target,rows=rows,signature=sig,family=family,c=c),file,compress=3)
            np.testing.assert_allclose(p,predict(joblib.load(file)["model"]),atol=1e-10,rtol=1e-10)
            result.append(p)
            models.append(dict(file=file.name,sha256=b.sha(file)))
    b.save(folder/"metadata.json",dict(signature=sig,models=models,fit_rows=len(rows),
        query_rows=len(query),idf_fit_only=True,model_reload_passed=True,seconds=time.monotonic()-started))
    temporary = folder/"predictions.tmp.npz"
    np.savez_compressed(temporary,p=np.stack(result),fit=rows,query=query,signature=sig)
    temporary.replace(path)
    b.emit("v70_fit",task=d["tasks"][k],partition=partition,seconds=round(time.monotonic()-started))
    return np.stack(result)


def initialize_worker():
    global WORKER_DATA, WORKER_VIEWS
    WORKER_DATA = b.build()
    WORKER_VIEWS = chemistry(WORKER_DATA)


def prepare_task(k):
    for part in base.P["splits"]:
        fit(WORKER_DATA,WORKER_VIEWS,k,part)
    return WORKER_DATA["tasks"][k]


def prepare(d, views):
    started = time.monotonic()
    b.save(OUT/"prepare_status.json",dict(status="RUNNING",pid=os.getpid(),signature=signature()))
    order = sorted(range(len(d["tasks"])),key=lambda k:np.sum((d["tid"]==k)&(d["split"]==0)))
    with ProcessPoolExecutor(max_workers=P["workers"],initializer=initialize_worker) as pool:
        for task in pool.map(prepare_task,order):
            b.emit("v70_task_prepared",task=task)
    b.save(OUT/"prepare_status.json",dict(status="COMPLETE",signature=signature(),seconds=time.monotonic()-started))


def select(d, views):
    assert json.loads((OUT/"prepare_status.json").read_text())["status"] == "COMPLETE"
    parent = json.loads((base.OUT/"selection.json").read_text())
    assert parent["signature"] == base.signature()
    records = []
    for k, prior in enumerate(parent["tasks"]):
        metric = prior["metric"]
        data = []
        for part in base.P["splits"]:
            query, old, sources = base.source_predictions(d,k,part)
            old = base.apply(metric,old,sources,prior["recipe"])
            if prior["task"] == "ppbr_az":old = old.clip(0,100)
            data.append((query,old,fit(d,views,k,part)))
        def evaluate(recipe):
            gains=[]
            for query,old,p in data:
                new=old.copy()
                for entry in recipe:new=b.blend(metric,new,p[entry["variant"]],entry["weight"])
                gains.append(b.gain(metric,d["y"][query],old,new))
            objective=P["selection"]["cv_weight"]*np.mean(gains[:3])+P["selection"]["valid_weight"]*gains[3]-P["selection"]["fold_sd_penalty"]*np.std(gains[:3])
            return dict(gains=gains,objective=float(objective),cv_mean=float(np.mean(gains[:3])))
        recipe=[];current=evaluate(recipe)
        for _ in range(P["selection"]["rounds"]):
            candidates=[]
            for variant in range(len(P["kernels"])*len(P["c_values"])):
                for weight in P["selection"]["weights"]:
                    proposed=recipe+[dict(variant=variant,weight=weight)]
                    candidates.append((evaluate(proposed),proposed))
            best,proposed=max(candidates,key=lambda r:r[0]["objective"])
            if best["objective"]<=current["objective"]+P["selection"]["minimum_objective_step"]:break
            recipe,current=proposed,best
        records.append(dict(task=prior["task"],metric=metric,recipe=recipe,development=current))
        b.emit("v70_selection",**records[-1])
    b.save(OUT/"selection.json",dict(status="FROZEN",signature=signature(),parent_signature=base.signature(),
        parent_selection_sha256=b.sha(base.OUT/"selection.json"),tasks=records,
        official_valid_used_for_tuning=True,independent_generalization_estimate=False))


def export(d, views):
    selection=json.loads((OUT/"selection.json").read_text())
    assert selection["signature"]==signature()
    assert selection["parent_selection_sha256"]==b.sha(base.OUT/"selection.json")
    parent=json.loads((base.OUT/"upload_release.json").read_text())
    source=ROOT/parent["file"]
    assert b.sha(source)==parent["sha256"]
    with source.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.reader(stream))
    for k,record in enumerate(selection["tasks"]):
        if not record["recipe"]:continue
        p=fit(d,views,k,"final")
        pos=[i for i,row in enumerate(rows[1:],1) if row[0]==record["task"]]
        new=np.array([float(rows[i][4]) for i in pos])
        for entry in record["recipe"]:new=b.blend(record["metric"],new,p[entry["variant"]],entry["weight"])
        if record["task"]=="ppbr_az":new=new.clip(0,100)
        if record["metric"] in ("AUROC","AUPRC"):new=new.clip(0,1)
        for i,value in zip(pos,new):rows[i][4]=repr(float(value))
    date=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    folder=ROOT/"admet_submissions/v70_sparse_aggressive"
    folder.mkdir(parents=True,exist_ok=True)
    file=folder/f"我从山北来带着兰花草-小分子药物预测技术创新赛-{date}.csv"
    buffer=io.StringIO(newline="");csv.writer(buffer,lineterminator="\r\n").writerows(rows)
    raw=b"\xef\xbb\xbf"+buffer.getvalue().encode("utf-8")
    if file.exists():assert file.read_bytes()==raw
    else:file.write_bytes(raw)
    b.save(OUT/"release.json",dict(status="AWAITING_INDEPENDENT_REPLAY",signature=signature(),
        file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),parent_file=parent["file"],
        parent_sha256=parent["sha256"],changed_tasks=[r["task"] for r in selection["tasks"] if r["recipe"]],
        external_data=False,pretrained_weights=False,official_score=None))
    b.emit("v70_export",file=str(file),sha256=b.sha(file))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","select","export"])
    args=parser.parse_args()
    assert os.name!="nt","Fits run on the authorized remote server"
    d=b.build();views=chemistry(d)
    {"prepare":prepare,"select":select,"export":export}[args.stage](d,views)
