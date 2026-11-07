"""Independently reload final trees, replay fusion, and apply the current upload template."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v69_aggressive as model
from verify_submission import read, verify


def main():
    d=b.build()
    release=json.loads((model.OUT/"release.json").read_text(encoding="utf-8"))
    selection=json.loads((model.OUT/"selection.json").read_text(encoding="utf-8"))
    assert release["signature"]==selection["signature"]==model.signature()
    source=ROOT/json.loads((ROOT/"admet_model_outputs_v68/release.json").read_text())["file"]
    assert b.sha(source)==release["baseline_sha256"]
    file=ROOT/release["file"]
    baseline,exported=read(source),read(file)
    checks=[]
    for k,record in enumerate(selection["tasks"]):
        positions=[i for i,row in enumerate(baseline[1:],1) if row[0]==record["task"]]
        if not record["recipe"]:
            assert all(baseline[i]==exported[i] for i in positions)
            continue
        rows,query=model.splits(d,k,"final")
        sources={}
        for family in ("shared","gated"):
            sources[family]=model.neural(d,query,"final",family)
        import admet_v68_kernel as kernel
        # Kernel save/load replay is also checked independently for newly final-fitted tasks.
        dest=kernel.OUT/record["task"]/"final"
        with np.load(dest/"predictions.npz",allow_pickle=False) as z:
            stored=z["p"].copy()
            np.testing.assert_array_equal(z["fit"],rows)
            np.testing.assert_array_equal(z["query"],query)
            assert str(z["signature"])==kernel.signature()
        import joblib
        j=0
        for family,(_,matrix) in enumerate(kernel.kernels(d,rows,query)):
            matrix=matrix.astype(np.float64)
            for c in kernel.P["c_values"]:
                checkpoint=dest/f"family{family}_C{c:g}.joblib"
                state=joblib.load(checkpoint)
                assert state["signature"]==kernel.signature()
                np.testing.assert_array_equal(state["rows"],rows)
                cls=record["metric"] in ("AUROC","AUPRC")
                p=state["model"].decision_function(matrix) if cls else state["model"].predict(matrix)
                if record["metric"]=="MAE":p=p*state["target"]["scale"]+state["target"]["center"]
                np.testing.assert_allclose(p,stored[j],atol=1e-9,rtol=1e-9)
                sources["kernel"+str(j)]=p;j+=1
                checks.append(dict(task=record["task"],kind="kernel",checkpoint_sha256=b.sha(checkpoint),
                    max_error=float(np.max(np.abs(p-stored[j-1])))))
        for profile in model.P["families"]:
            members=[]
            for seed in model.P["final_seeds"]:
                dest=model.OUT/record["task"]/"final"/profile["name"]/str(seed)
                metadata=json.loads((dest/"metadata.json").read_text(encoding="utf-8"))
                assert metadata["signature"]==model.signature()
                checkpoint=dest/"model.cbm"
                assert b.sha(checkpoint)==metadata["model_sha256"]
                with np.load(dest/"preprocessing.npz",allow_pickle=False) as z:
                    state={key:z[key] for key in z.files}
                np.testing.assert_array_equal(state["fit_molecules"],np.unique(d["mi"][rows]))
                _,x,_=model.features(d,rows,query,profile,state)
                net=(CatBoostClassifier if record["metric"] in ("AUROC","AUPRC") else CatBoostRegressor)()
                net.load_model(str(checkpoint))
                p=model.predict_model(net,x,record["metric"],metadata["target"])
                with np.load(dest/"predictions.npz",allow_pickle=False) as z:
                    assert str(z["signature"])==model.signature()
                    np.testing.assert_array_equal(z["fit"],rows)
                    np.testing.assert_array_equal(z["query"],query)
                    error=float(np.max(np.abs(p-z["p"])))
                    np.testing.assert_allclose(p,z["p"],atol=1e-10,rtol=1e-10)
                members.append(p)
                checks.append(dict(task=record["task"],kind="catboost",profile=profile["name"],seed=seed,
                    checkpoint_sha256=b.sha(checkpoint),max_error=error))
            sources[profile["name"]]=np.mean(members,0)
        old=np.array([float(baseline[i][4]) for i in positions])
        expected=model.apply(record["metric"],old,sources,record["recipe"])
        if record["task"]=="ppbr_az":expected=expected.clip(0,100)
        if record["metric"] in ("AUROC","AUPRC"):expected=expected.clip(0,1)
        np.testing.assert_allclose(expected,[float(exported[i][4]) for i in positions],atol=1e-9,rtol=1e-9)
        b.emit("v69_final_replay",task=record["task"],status="PASS")
    audit=verify(file,ROOT/"admet_official/submission_template.csv")
    assert audit["sha256"]==release["sha256"]
    # Actual platform validation requires IDs such as '5348064.0', as in the official template.
    template=ROOT/"admet_official/submission_template.csv"
    assert b.sha(template)=="8ee41589d2c98e7b93081c058b3f928ac09bea4b3ffc899fa13677051e9baa20"
    assert next(row for row in read(template)[1:] if row[:2]==["cyp2c9_veith","0"])[2]=="5348064.0"
    output=ROOT/"admet_submissions/v69_aggressive_platform_ids"/file.name
    subprocess.run([sys.executable,str(ROOT/"tools/repack_upload_template.py"),str(file),
                    "--template",str(template),"--output",str(output)],check=True,cwd=ROOT)
    upload=verify(output,template)
    b.save(model.OUT/"upload_release.json",dict(status="AWAITING_PLATFORM_EVALUATION",
        file=output.relative_to(ROOT).as_posix(),sha256=upload["sha256"],rows=upload["rows"],tasks=upload["tasks"],
        template_sha256=b.sha(template),predictions_exact=True,inference_sha256=release["sha256"],official_score=None))
    b.save(model.OUT/"model_replay.json",dict(status="PASS",checks=checks,sha256=release["sha256"],
        upload_sha256=upload["sha256"],exported_predictions_replayed=True,persisted_preprocessing_reloaded=True,
        official_valid_used_for_tuning=True,official_score=None))
    b.emit("v69_independent_replay",status="PASS",checkpoints=len(checks),upload_sha256=upload["sha256"])


if __name__=="__main__":main()
