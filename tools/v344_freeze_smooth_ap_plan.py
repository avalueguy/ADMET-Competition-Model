"""Freeze direct-AP experiment before computing any development metric."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    smoke = json.loads((ROOT / "admet_model_outputs_v340/smoke.json").read_text(encoding="utf-8"))
    assert smoke["status"] == "PASS" and not smoke["query_scores_computed"]
    assert not (ROOT / "admet_model_outputs_v340/development.json").exists()
    path = ROOT / "docs/v340_training_plan.json"
    assert not path.exists()
    plan = dict(status="FROZEN_BEFORE_DEVELOPMENT_SCORES",
        at_hong_kong=datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(), baseline="v313",
        official_best_user_reported=51.205, target_score=52.,
        source_sha256=sha(ROOT / "tools/v340_smooth_ap_retrieval.py"),
        baseline_adapter_sha256=sha(ROOT / "tools/v314_current_baseline.py"),
        runner_sha256=sha(ROOT / "tools/v314_v313_experiment_runner.py"),
        verifier_sha256=sha(ROOT / "tools/v341_verify_smooth_ap.py"), smoke_signature=smoke["signature"],
        tasks=["cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels", "cyp2c9_veith", "cyp2d6_veith", "cyp3a4_veith"],
        config=dict(neighbors=24, width=128, key_width=32, bins=8, epochs=100, minimum_steps=1200,
                    maximum_steps=2500, ap_temperature=.2, batch_size=256),
        seeds=[3041101, 3041102], projection_seed=3040001,
        method="New differentiable listwise AP surrogate .5 BCE+.5(1-softAP). Each positive's precision uses 1+sigmoid-smoothed count of higher-ranked positive/all rows, diagonal excluded, temperature.2. Uniform256 rows retain fit prevalence. Original v315 retrieval architecture, AdamW lr and optimizer update count, larger batch; train from scratch, no query-label stopping. No external data.",
        justification="Five official tasks use AUPRC; direct metric-aligned listwise objective differs from historical BCE/pairwise neural losses and tree LambdaMAP. Temperature/loss weights fixed from definition, not scored search. Does not retry CYP3A4 substrate or tune rejected SAM/DRO weights/seeds.",
        selection="Original v148/v314 max minimum fold0/1 gain, original blends.20/.45; freeze before fold2/valid, threshold>.003, no failed-confirmation fallback.",
        confirmation="All four positive, mean>.003, both500 paired scaffold bootstrap90% lower bounds>0.",
        smoke="PASS independent scalar AP loss, finite-difference gradient and exact hard-ranking AP limit; degenerate classes; finite CUDA gradients; feature label isolation; saved-model replay; reversed-query consistency. No query metric read.",
        protected_tasks=["ppbr_az", "lipophilicity_astrazeneca", "cyp3a4_substrate_carbonmangels"],
        test_labels_used=False, external_data=False, automatic_platform_submission=False,
        limitations="Repeated public labels and approximate inherited historical OOF remain. New surrogate may not improve held-out AP. No claim of official score gain or52.")
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan))


if __name__ == "__main__":
    main()
