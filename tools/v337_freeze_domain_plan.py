"""Freeze group-DRO plan after mechanics-only smoke, before development scores."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    smoke = json.loads((ROOT / "admet_model_outputs_v333/smoke.json").read_text(encoding="utf-8"))
    assert smoke["status"] == "PASS" and not smoke["query_scores_computed"]
    assert not (ROOT / "admet_model_outputs_v333/development.json").exists()
    path = ROOT / "docs/v333_training_plan.json"
    assert not path.exists()
    plan = dict(status="FROZEN_BEFORE_DEVELOPMENT_SCORES",
        at_hong_kong=datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(), baseline="v313",
        official_best_user_reported=51.205, target_score=52.,
        source_sha256=sha(ROOT / "tools/v333_chemical_domain_dro.py"),
        original_source_sha256=sha(ROOT / "tools/v315_rank_label_retrieval.py"),
        baseline_adapter_sha256=sha(ROOT / "tools/v314_current_baseline.py"),
        runner_sha256=sha(ROOT / "tools/v314_v313_experiment_runner.py"),
        verifier_sha256=sha(ROOT / "tools/v334_verify_domain_dro.py"), smoke_signature=smoke["signature"],
        tasks=["herg", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels"],
        config=dict(neighbors=24, width=128, key_width=32, bins=8, epochs=100,
                    minimum_steps=1200, maximum_steps=2500, domains=8, dro_eta=.01, dro_fraction=.5),
        seeds=[3041101, 3041102], projection_and_cluster_seed=3040001,
        method="Fit-only scaffold mean normalized4096 count fingerprints clustered8 by KMeans(n_init1,max_iter100). No query structure used for clusters. Balanced batch16 rows/domain. Dynamic exponentiated group-BCE adversary eta.01, half fit-row-prior ERM plus half DRO plus .25 pair loss. Same v315 network, AdamW duration and seeds from scratch.",
        selection="Original v148/v314 max minimum fold0/1 gain; original fixed blends.20/.45. Freeze before fold2/valid. No failed-confirmation fallback.",
        confirmation="All four positive, mean gain>.003, both500 paired scaffold bootstrap90% lower bounds>0.",
        smoke="PASS independent NumPy loss/gradient/adversary references; degenerate classes; scaffold integrity; fit-only grouping; balanced batch; finite CUDA gradients; nonfit-label isolation; saved-model replay. No query metrics.",
        protected_tasks=["ppbr_az", "lipophilicity_astrazeneca", "cyp3a4_substrate_carbonmangels"],
        test_labels_used=False, external_data=False, automatic_platform_submission=False,
        limitations="Public labels reused; historical OOF approximate. A distinct training mechanism is not proof of official improvement. Current best51.205, no claim of52.")
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan))


if __name__ == "__main__":
    main()
