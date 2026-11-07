"""Finish diagnostics and archive each verified remote release as soon as it is ready."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/"releases/aggressive_completion_status.json"


def save(value):
    STATE.parent.mkdir(exist_ok=True)
    temp=STATE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    temp.replace(STATE)


def main():
    for version in (69,70):
        save(dict(status="WAITING",version=version,official_score=None))
        path=ROOT/f"admet_model_outputs_v{version}/pipeline_status.json"
        while True:
            if path.exists():
                state=json.loads(path.read_text())
                if state["status"]=="FAILED":raise RuntimeError(f"v{version} pipeline failed: {state}")
                if state["status"]=="COMPLETE":break
            time.sleep(30)
        for stage in ("report","package"):
            save(dict(status="RUNNING",version=version,stage=stage,official_score=None))
            name=f"tools/report_v{version}_development.py" if stage=="report" else f"tools/package_v{version}_release.py"
            subprocess.run([sys.executable,"-u",name],cwd=ROOT,check=True)
    save(dict(status="COMPLETE",versions=[69,70],official_score=None))


if __name__=="__main__":
    try:main()
    except BaseException as error:
        save(dict(status="FAILED",error=str(error),official_score=None));raise
