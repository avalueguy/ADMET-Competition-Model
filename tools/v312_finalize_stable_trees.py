"""Wait for the specific v310 process, then independently verify once."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    pid = int(sys.argv[1])
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        # Refuse PID reuse rather than following an unrelated process.
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        assert b"v310_stable_differentiable_trees.py" in cmdline, "Training PID no longer identifies v310"
        time.sleep(15)
    status = ROOT / "docs/v310_final_status.json"
    if not status.exists():
        raise RuntimeError("v310 exited without a final status; inspect v310_training.log")
    data = json.loads(status.read_text())
    assert data["status"] in ("COMPLETE_NO_CONFIRMED_NEW_MODEL", "REQUIRES_INDEPENDENT_DELIVERY_VERIFICATION")
    subprocess.run([sys.executable, "-u", str(ROOT / "tools/v311_verify_stable_trees.py")], cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
