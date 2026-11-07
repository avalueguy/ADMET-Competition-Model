"""Create explicit v324 verification/finalization from the audited v316 workflow."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / "tools/v316_verify_rank_retrieval.py").read_text(encoding="utf-8")
    for before, after in (("v315_rank_label_retrieval", "v324_scaffold_manifold_mixup"),
                          ("v315", "v324"), ("v316", "v325"), ("v317_finalize_rank_retrieval", "v327_finalize_manifold_mixup"),
                          ("learned_retrieval", "manifold_mixup")):
        source = source.replace(before, after)
    source = source.replace('ROOT / "tools/v314_current_baseline.py",',
                            'ROOT / "tools/v315_rank_label_retrieval.py", ROOT / "tools/v314_current_baseline.py",')
    target = ROOT / "tools/v325_verify_manifold_mixup.py"
    assert not target.exists()
    target.write_text(source, encoding="utf-8")
    source = (ROOT / "tools/v317_finalize_rank_retrieval.py").read_text(encoding="utf-8")
    source = source.replace("v315_rank_label_retrieval", "v324_scaffold_manifold_mixup").replace("v315", "v324")
    source = source.replace("v316_verify_rank_retrieval", "v325_verify_manifold_mixup")
    target = ROOT / "tools/v327_finalize_manifold_mixup.py"
    assert not target.exists()
    target.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
