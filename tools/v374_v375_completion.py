"""Record completion only after both experiment runners and verifiers exit."""
from pathlib import Path
import json
import os
import subprocess
import time


def main():
    roots = [Path(f"/root/autodl-tmp/mae_experts_v{v}") for v in (374, 375)]
    statuses = [json.loads((p / "run_status.json").read_text()) for p in roots]
    assert all(x["stage"] == "EVIDENCE_READY" for x in statuses)
    matching = []
    names = tuple(f"v{v}_{name}.py" for v in (374, 375) for name in ("run_experts", "verify_experts"))
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if state != "Z" and any(name in args for name in names):
            matching.append(dict(pid=int(proc.name), command=args))
    assert not matching, matching
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"], text=True)
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
    assert not compute.strip(), compute
    receipt = dict(no_training_processes_running=True, experiment_pids=[x["pid"] for x in statuses],
                   scope="Both v374/v375 runners and verifiers absent; no GPU compute processes.",
                   at_unix=time.time(), gpu=gpu, gpu_compute_processes=compute,
                   statuses={p.name: x["stage"] for p, x in zip(roots, statuses)})
    for p in roots:
        (p / "completion_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
