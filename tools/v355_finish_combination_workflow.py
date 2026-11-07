"""Wait for the authorized confirmation process, then run frozen delivery once."""
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
        assert b"v350_run_combination_confirmation.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
        time.sleep(15)
    result = ROOT / "admet_model_outputs_v349/confirmation.json"
    assert result.exists(), "Confirmation exited without completion; inspect v349_confirmation.log"
    assert json.loads(result.read_text())["status"] == "COMPLETE"
    subprocess.run([sys.executable, "-u", "tools/v354_deliver_frozen_combinations.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
