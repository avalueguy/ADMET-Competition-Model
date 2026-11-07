"""Reload sparse kernel models, rebuild their matrices and replay the v70 CSV."""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v70_sparse as model
from verify_submission import read,verify


def main():
    d=b.build();views=model.chemistry(d)
    release=json.loads((model.OUT/"release.json").read_text())
    selection=json.loads((model.OUT/"selection.json").read_text())
    assert release["signature"]==selection["signature"]==model.signature()
    parent=json.loads((model.base.OUT/"model_replay.json").read_text())
    assert parent["status"]=="PASS" and parent["upload_sha256"]==release["parent_sha256"]
    assert b.sha(ROOT/release["parent_file"])==release["parent_sha256"]
    baseline,exported=read(ROOT/release["parent_file"]),read(ROOT/release["file"])
    checks=[]
    for k,record in enumerate(selection["tasks"]):
        assert record["task"]==d["tasks"][k]
        positions=[i for i,row in enumerate(baseline[1:],1) if row[0]==record["task"]]
        if not record["recipe"]:
            assert all(baseline[i]==exported[i] for i in positions)
            continue
        rows,query=model.base.splits(d,k,"final")
        folder=model.OUT/record["task"]/"final"
        with np.load(folder/"predictions.npz",allow_pickle=False) as z:
            assert str(z["signature"])==model.signature()
            np.testing.assert_array_equal(z["fit"],rows)
            np.testing.assert_array_equal(z["query"],query)
            stored=z["p"].copy()
        predictions=[]
        metadata=json.loads((folder/"metadata.json").read_text())
        model_hashes={r["file"]:r["sha256"] for r in metadata["models"]}
        for family,spec in enumerate(model.P["kernels"]):
            _,matrix=model.matrices(d,views,rows,query,spec)
            for c in model.P["c_values"]:
                checkpoint=folder/f"family{family}_C{c:g}.joblib"
                assert b.sha(checkpoint)==model_hashes[checkpoint.name]
                state=joblib.load(checkpoint)
                assert state["signature"]==model.signature()
                assert state["family"]==family and state["c"]==c
                np.testing.assert_array_equal(state["rows"],rows)
                cls=record["metric"] in ("AUROC","AUPRC")
                p=state["model"].decision_function(matrix) if cls else state["model"].predict(matrix)
                if record["metric"]=="MAE":p=p*state["target"]["scale"]+state["target"]["center"]
                error=float(np.max(np.abs(p-stored[len(predictions)])))
                np.testing.assert_allclose(p,stored[len(predictions)],atol=1e-9,rtol=1e-9)
                predictions.append(p)
                checks.append(dict(task=record["task"],family=family,c=c,max_error=error,
                    checkpoint_sha256=b.sha(checkpoint)))
        expected=np.array([float(baseline[i][4]) for i in positions])
        for entry in record["recipe"]:
            expected=b.blend(record["metric"],expected,predictions[entry["variant"]],entry["weight"])
        if record["task"]=="ppbr_az":expected=expected.clip(0,100)
        if record["metric"] in ("AUROC","AUPRC"):expected=expected.clip(0,1)
        np.testing.assert_allclose(expected,[float(exported[i][4]) for i in positions],atol=1e-9,rtol=1e-9)
        b.emit("v70_final_replay",task=record["task"],status="PASS",models=len(predictions))
    source=ROOT/release["file"]
    assert b.sha(source)==release["sha256"]
    template=ROOT/"admet_official/submission_template.csv"
    assert b.sha(template)=="8ee41589d2c98e7b93081c058b3f928ac09bea4b3ffc899fa13677051e9baa20"
    output=ROOT/"admet_submissions/v70_platform_ids"/source.name
    subprocess.run([sys.executable,str(ROOT/"tools/repack_upload_template.py"),str(source),
        "--template",str(template),"--output",str(output)],check=True,cwd=ROOT)
    audit=verify(output,template)
    b.save(model.OUT/"upload_release.json",dict(status="LOCALLY_VERIFIED_AWAITING_PLATFORM_EVALUATION",
        file=output.relative_to(ROOT).as_posix(),sha256=audit["sha256"],rows=audit["rows"],tasks=audit["tasks"],
        template_sha256=b.sha(template),official_score=None,automatically_uploaded=False))
    b.save(model.OUT/"model_replay.json",dict(status="PASS",checks=checks,
        sha256=release["sha256"],upload_sha256=audit["sha256"],parent_sha256=release["parent_sha256"],
        exported_predictions_replayed=True,idf_recomputed_from_fit_only=True,
        official_valid_used_for_tuning=True,independent_generalization_estimate=False,official_score=None))
    b.emit("v70_independent_replay",status="PASS",models=len(checks),sha256=audit["sha256"])


if __name__=="__main__":main()
