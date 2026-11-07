"""Detached, resumable official-only CUDA pipeline with explicit completion state."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "admet_model_outputs_v67"


def save(value):
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT / "pipeline_status.json.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT / "pipeline_status.json")


def main():
    started = time.time()
    base = dict(pid=os.getpid(), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                external_data=False, pretrained_weights=False, automatically_uploaded=False)
    for stage, command in [
        ("data", ["admet_v67_data.py"]),
        ("smoke", ["admet_v67_train.py", "smoke"]),
        ("training", ["admet_v67_train.py", "all"]),
    ]:
        save(dict(**base, status="RUNNING", stage=stage, elapsed_seconds=int(time.time()-started)))
        result = subprocess.run([sys.executable, "-u", *command], cwd=ROOT)
        if result.returncode:
            save(dict(**base, status="FAILED", stage=stage, exit_code=result.returncode,
                      elapsed_seconds=int(time.time()-started)))
            raise SystemExit(result.returncode)
    release = json.loads((OUT / "release.json").read_text(encoding="utf-8"))
    file = ROOT / release["file"]
    subprocess.run([sys.executable, str(ROOT / "tools/verify_submission.py"), str(file)], check=True, cwd=ROOT)
    save(dict(**base, status="COMPLETE", stage="verified_export", elapsed_seconds=int(time.time()-started),
              csv=str(file), sha256=release["sha256"], official_score=None))


if __name__ == "__main__":
    main()
