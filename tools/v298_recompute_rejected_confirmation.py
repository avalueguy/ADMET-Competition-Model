"""Recalculate frozen v297 Caco2 confirmation from saved prediction arrays only."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v297_pair_delta_regression as model

def main():
    d = model.load()
    confirmation = json.loads((model.OUT / "confirmation.json").read_text())
    frozen = json.loads((model.OUT / "frozen_selection.json").read_text())
    assert confirmation["status"] == "COMPLETE"
    checks = []
    for record in confirmation["records"]:
        choice = next(r for r in frozen["selected"] if r["task"] == record["task"])
        k = d["tasks"].index(record["task"])
        recomputed = model.runner.confirm(d, k, model.predict, choice)
        assert recomputed == record
        checks.append(dict(task=record["task"], all_four_metrics_exact=True,
                           bootstrap_intervals_exact=True, accepted=record["accepted"]))
    result = dict(status="PASS", signature=model.sig(), checks=checks,
        model_checkpoints_reloaded=False, saved_prediction_gate_recomputation_only=True,
        at_utc=datetime.now(timezone.utc).isoformat(), accepted=confirmation["accepted"])
    model.b.save(ROOT / "docs/v298_confirmation_recomputation.json", result)
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
