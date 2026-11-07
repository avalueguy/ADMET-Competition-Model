"""Resume the kernel release after its official-only neural prerequisite finishes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "admet_model_outputs_v68"


def save(stage, status="RUNNING", **details):
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT / "pipeline_status.json.tmp"
    temporary.write_text(json.dumps(dict(pid=os.getpid(), stage=stage, status=status,
                          automatically_uploaded=False, official_score=None, **details), indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT / "pipeline_status.json")


def main():
    started = time.monotonic()
    save("waiting_for_prerequisites")
    while True:
        path = ROOT / "admet_model_outputs_v67/pipeline_status.json"
        neural = json.loads(path.read_text()) if path.exists() else {}
        if neural.get("status") == "FAILED":
            raise RuntimeError("v67 prerequisite failed; inspect its logs")
        if neural.get("status") == "COMPLETE" and (OUT / "prepare_completed.json").exists():
            break
        if time.monotonic() - started > 1800:
            raise TimeoutError("Prerequisites incomplete after 30 minutes; no release certified")
        time.sleep(10)
    for stage, arguments in [
        ("verify_neural_checkpoints", ["tools/verify_v67_models.py"]),
        ("freeze_kernel_selection_and_export", ["admet_v68_kernel.py", "finish"]),
        ("verify_kernel_checkpoints", ["tools/verify_v68_models.py"]),
    ]:
        save(stage)
        subprocess.run([sys.executable, "-u", *arguments], cwd=ROOT, check=True)
    release = json.loads((OUT / "release.json").read_text(encoding="utf-8"))
    save("verified_export", "COMPLETE", sha256=release["sha256"],
         csv=release["file"], elapsed_seconds=round(time.monotonic()-started))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save("failure", "FAILED", error=str(error))
        raise
