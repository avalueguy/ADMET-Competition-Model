"""Create explicit SAM verification workflow from the audited retrieval verifier."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / "tools/v316_verify_rank_retrieval.py").read_text(encoding="utf-8")
    for before, after in (("v315_rank_label_retrieval", "v329_sharpness_retrieval"),
                          ("v315", "v329"), ("v316", "v330"),
                          ("v317_finalize_rank_retrieval", "v331_finalize_sharpness"),
                          ("learned_retrieval", "sharpness_retrieval")):
        source = source.replace(before, after)
    source = source.replace('ROOT / "tools/v314_current_baseline.py",',
                            'ROOT / "tools/v304_learned_label_retrieval.py", ROOT / "tools/v314_current_baseline.py",')
    source = source.replace('    d = model.load()\n', '    model.optimizer_reference()\n    d = model.load()\n')
    target = ROOT / "tools/v330_verify_sharpness.py"
    assert not target.exists()
    target.write_text(source, encoding="utf-8")
    source = (ROOT / "tools/v317_finalize_rank_retrieval.py").read_text(encoding="utf-8")
    source = source.replace("v315_rank_label_retrieval", "v329_sharpness_retrieval").replace("v315", "v329")
    source = source.replace("v316_verify_rank_retrieval", "v330_verify_sharpness")
    target = ROOT / "tools/v331_finalize_sharpness.py"
    assert not target.exists()
    target.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
