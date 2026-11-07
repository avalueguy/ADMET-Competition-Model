"""Wait for the specific v318 process, then independently verify once."""
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
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        assert b"v318_neural_tangent_regression.py" in command, "Training PID no longer identifies v318"
        time.sleep(15)
    status = ROOT / "docs/v318_final_status.json"
    if not status.exists():
        raise RuntimeError("v318 exited without a final status; inspect v318_training.log")
    data = json.loads(status.read_text())
    assert data["status"] in ("COMPLETE_NO_CONFIRMED_NEW_MODEL", "REQUIRES_INDEPENDENT_DELIVERY_VERIFICATION")
    subprocess.run([sys.executable, "-u", str(ROOT / "tools/v319_verify_neural_tangent.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
