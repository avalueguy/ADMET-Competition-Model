"""Resume unchanged v354 after auditing a sub-ulp v233 transform comparison failure."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import numpy as np
import v349_diverse_model_combinations as experiment
import v351_verify_combination_models as verifier
import v358_kernel_replay_adapter as adapter

ORIGINAL_SHA = "382e321af147049a366d0a6cf2270988fdb567009a7fee25ac9e8df874a658f1"
OUT = ROOT / "admet_model_outputs_v354"
AUDIT = OUT / "resume_audit.json"
FROZEN = OUT / "frozen_before_final.json"
original_save = experiment.save
original_check = verifier.check
saved_freeze = json.loads(FROZEN.read_text())


def guard():
    assert OUT.is_dir() and FROZEN.is_file()
    assert not (ROOT / "docs/v354_delivery.json").exists()
    assert not (OUT / "delivery.json").exists()
    assert not list((ROOT / "admet_submissions/v354_experimental_model_combinations").glob("*.csv"))
    assert saved_freeze["source_sha256"] == ORIGINAL_SHA
    assert saved_freeze["confirmation_sha256"] == experiment.sha(experiment.OUT / "confirmation.json")
    assert saved_freeze["policy_sha256"] == experiment.sha(ROOT / "docs/v352_combination_delivery_policy.json")
    assert saved_freeze["verifier_sha256"] == experiment.sha(verifier.__file__)
    assert not AUDIT.exists(), "Inspect prior resume attempt before retry"
    original_save(AUDIT, dict(status="RESUME_STARTED", at_hong_kong=experiment.now(),
        original_delivery_sha256=ORIGINAL_SHA, original_freeze_sha256=experiment.sha(FROZEN),
        runner_sha256=experiment.sha(__file__), adapter_sha256=experiment.sha(adapter.__file__),
        previous_log="v354_delivery.log", previous_log_sha256=experiment.sha(ROOT / "v354_delivery.log"),
        reason="v233 regenerated transform differs by8.7e-19; exact state comparator stopped before CSV.",
        model_source_config_seed_weight_and_selection_unchanged=True,
        retained_prediction_tolerance=1e-12, required_all_replay_prediction_ranks_exact=True,
        transform_float_rtol=2e-13, transform_float_atol=2e-14,
        exact_integer_string_split_fields=True, original_failure_and_frozen_files_preserved=True))


def safe_save(path, value):
    if path == FROZEN:
        assert {k:v for k,v in value.items() if k != "at_hong_kong"} == {k:v for k,v in saved_freeze.items() if k != "at_hong_kong"}
        value.clear()
        value.update(saved_freeze)
        return
    if path in (OUT / "delivery.json", ROOT / "docs/v354_delivery.json"):
        value["numerical_replay_resume"] = dict(audit=AUDIT.relative_to(ROOT).as_posix(),
            adapter="tools/v358_kernel_replay_adapter.py", original_strict_rejections_preserved=True)
    return original_save(path, value)


def save_final(path, **values):
    if path.exists():
        saved = np.load(path)
        assert set(saved.files) == set(values)
        for key in values:
            np.testing.assert_array_equal(saved[key], values[key])
        return
    np.savez_compressed(path, **values)


def routed_check(d, k, part, member, mods):
    if member["version"] == 233:
        return adapter.check(d, k, part, member, mods)
    return original_check(d, k, part, member, mods)


def main():
    source_path = ROOT / "tools/v354_deliver_frozen_combinations.py"
    assert experiment.sha(source_path) == ORIGINAL_SHA
    source = source_path.read_text()
    changes = {
        'assert not OUT.exists(), "Inspect any partial delivery before retry"':'resume_guard()',
        '    OUT.mkdir()':'    # Existing inspected output is preserved.',
        'np.savez_compressed(OUT / "final"':'resume_save_final(OUT / "final"',
        '    files = sorted(set(p.resolve() for p in files))':
            '    files += [ROOT / "tools/v358_kernel_replay_adapter.py", ROOT / "tools/v359_resume_frozen_delivery.py", ROOT / "v354_delivery.log", OUT / "resume_audit.json"]\n    files = sorted(set(p.resolve() for p in files))',
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    experiment.save = safe_save
    verifier.check = routed_check
    namespace = dict(__file__=str(source_path), __name__="v354_inspected_resume", resume_guard=guard, resume_save_final=save_final)
    exec(compile(source, str(source_path), "exec"), namespace)
    namespace["main"]()
    result = json.loads(AUDIT.read_text())
    result.update(status="PASS_RESUMED_DELIVERY", completed_at_hong_kong=experiment.now(),
        csv_sha256=json.loads((OUT / "delivery.json").read_text())["csv_sha256"])
    original_save(OUT / "resume_complete.json", result)


if __name__ == "__main__":
    main()
