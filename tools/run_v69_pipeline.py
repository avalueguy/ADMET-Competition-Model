"""Durable aggressive search with accurate completion and failure records."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"admet_model_outputs_v69"


def state(stage,status="RUNNING",**details):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"pipeline_status.json.tmp"
    path.write_text(json.dumps(dict(stage=stage,status=status,pid=os.getpid(),official_score=None,
        automatically_uploaded=False,**details),indent=2)+"\n",encoding="utf-8")
    path.replace(OUT/"pipeline_status.json")


def main():
    started=time.monotonic()
    for stage,args in [("trees_and_kernels",["admet_v69_aggressive.py","prepare"]),
                       ("selection",["admet_v69_aggressive.py","select"]),
                       ("final_fit_export",["admet_v69_aggressive.py","export"]),
                       ("independent_model_replay",["tools/verify_v69_models.py"])]:
        state(stage,elapsed_seconds=round(time.monotonic()-started))
        subprocess.run([sys.executable,"-u",*args],cwd=ROOT,check=True)
    release=json.loads((OUT/"release.json").read_text())
    state("verified_export","COMPLETE",elapsed_seconds=round(time.monotonic()-started),sha256=release["sha256"])


if __name__=="__main__":
    try:main()
    except BaseException as error:
        state("failure","FAILED",error=str(error));raise
