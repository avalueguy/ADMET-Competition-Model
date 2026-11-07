"""Prepare CPU sparse kernels, then finish after a verified v69 parent is available."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"admet_model_outputs_v70"


def state(stage,status="RUNNING",**details):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"pipeline_status.json.tmp"
    path.write_text(json.dumps(dict(stage=stage,status=status,pid=os.getpid(),
        official_score=None,automatically_uploaded=False,**details),indent=2)+"\n",encoding="utf-8")
    path.replace(OUT/"pipeline_status.json")


def run(args):
    subprocess.run([sys.executable,"-u",*args],cwd=ROOT,check=True)


def main():
    started=time.monotonic()
    state("sparse_kernel_development")
    run(["admet_v70_sparse.py","prepare"])
    state("waiting_for_verified_v69")
    while True:
        path=ROOT/"admet_model_outputs_v69/pipeline_status.json"
        if path.exists():
            parent=json.loads(path.read_text())
            if parent["status"]=="FAILED":raise RuntimeError("v69 parent failed; repair parent before resuming")
            if parent["status"]=="COMPLETE":break
        time.sleep(30)
    for stage,args in [("selection",["admet_v70_sparse.py","select"]),
                       ("final_fit_export",["admet_v70_sparse.py","export"]),
                       ("independent_model_replay",["tools/verify_v70_models.py"])]:
        state(stage,elapsed_seconds=round(time.monotonic()-started))
        run(args)
    release=json.loads((OUT/"upload_release.json").read_text())
    state("verified_export","COMPLETE",elapsed_seconds=round(time.monotonic()-started),sha256=release["sha256"])


if __name__=="__main__":
    try:main()
    except BaseException as error:
        state("failure","FAILED",error=str(error));raise
