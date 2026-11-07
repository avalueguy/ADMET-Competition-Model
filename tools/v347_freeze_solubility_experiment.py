"""Record authorized experimental scope, preserving the original strict rejection."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = ROOT / "docs/v346_experimental_plan.json"
    assert not path.exists()
    frozen = json.loads((ROOT / "admet_model_outputs_v329/frozen_selection.json").read_text(encoding="utf-8"))
    confirmation = json.loads((ROOT / "admet_model_outputs_v329/confirmation.json").read_text(encoding="utf-8"))
    task = "solubility_aqsoldb"
    choice = next(r for r in frozen["selected"] if r["task"] == task)
    rejected = next(r for r in confirmation["records"] if r["task"] == task)
    assert choice["weight"] == .2 and rejected["accepted"] is False
    assert min(r["gain"] for r in rejected["confirmation"]) > 0
    plan = dict(status="FROZEN_BEFORE_FINAL_EXPERIMENTAL_TRAINING",
        at_hong_kong=datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        user_authorized_experimental_csv=True,
        authorization="Prior explicit user choice: 另做实验 CSV 手动测试. Latest user steering: 只有有提升就行. Manual experimental delivery remains authorized; no guarantee of gain or automatic submission.",
        goal="Seek any positive official score change over current best v313, rather than52",
        baseline="v313", official_best=51.205,
        baseline_csv_sha256="681e9d23100c7ea39b4e6003dd6de4fac738bc467254782b7b8c4b7bc1d2941b",
        source_sha256=sha(ROOT / "tools/v346_experimental_solubility.py"),
        model_source_sha256=sha(ROOT / "tools/v329_sharpness_retrieval.py"),
        verifier_sha256=sha(ROOT / "tools/v330_verify_sharpness.py"),
        runner_sha256=sha(ROOT / "tools/v314_v313_experiment_runner.py"),
        task=task, weight=.2, choice=choice, strict_confirmation=rejected,
        strict_gate_pass=False, strict_rejection_unchanged=True, no_retuning=True,
        frozen_seeds=[3041101, 3041102],
        selection="Retrospective single-task experiment. Both solubility and hERG have four positive points; solubility has supported fold2 CI and substantially narrower valid uncertainty. hERG is not added. No failed v321 predictions, confirmation-driven blend sweep, or sample edits.",
        training="Same immutable v329 SAM architecture, optimizer, seeds and duration; final fit includes permitted official training and validation rows; only final predictions computed. Separate output directory preserves all scored folds.",
        unchanged_tasks=21, manual_submission_only=True, platform_submission=False,
        test_labels_used=False, external_data=False, official_gain_established=False,
        limitations="Public labels reused, approximate historical OOF, valid confidence interval crosses zero. Candidate may regress on official test. File is experimental and does not supersede v313 until positive user feedback.")
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan))


if __name__ == "__main__":
    main()
