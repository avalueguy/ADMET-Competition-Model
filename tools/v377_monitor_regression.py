"""Read-only compact progress for the two fixed-budget remote experiments."""
from pathlib import Path
import json
import subprocess
import time


def main():
    result = {"at_unix": time.time(), "experiments": {}}
    for version in (377, 378):
        root = Path(f"/root/autodl-tmp/regression9_v{version}")
        item = {}
        for name in ("run_status", "launch", "queue_status"):
            path = root / (name + ".json")
            if path.exists():
                item[name] = json.loads(path.read_text())
        path = root / "training.log"
        events = []
        if path.exists():
            for line in path.read_text(errors="replace").splitlines():
                if line.startswith('{"stage":'):
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        done = [e for e in events if e.get("stage") == "curve_job_done"]
        if done:
            item["development_curves"] = done[-1]
        if events:
            event = events[-1]
            item["latest_event"] = {k: v for k, v in event.items()
                                    if k in ("stage", "task", "part", "base", "seed", "steps", "seconds")}
        prefit = root / 'parallel_prefit_results.json'
        if prefit.exists():
            parallel = json.loads(prefit.read_text())
            item['parallel_prefit'] = dict(completed=len(parallel['results']), total=parallel['total'],
                                          failed=sum(x['exit_code'] != 0 for x in parallel['results']))
        if (root/'parallel_prefit_launch.json').exists():
            item['scheduling'] = 'original driver suspended until independent unchanged curve jobs finish' if not (root/'parallel_prefit_complete.json').exists() else 'original driver resumed'
        item["curve_states_saved"] = len(list((root / "outputs/curves").glob("*/*/*/state.joblib")))
        item["prediction_packages_saved"] = len(list((root / "outputs/tasks").glob("*/*/*/predictions.npz")))
        if "launch" in item:
            proc = Path('/proc') / str(item["launch"]["pid"])
            item["runner_pid_exists"] = proc.exists()
        result["experiments"][str(version)] = item
    result["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
    path=Path('/root/autodl-tmp/regression9_progress.json')
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(path)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
