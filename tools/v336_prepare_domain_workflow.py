"""Create chemical-domain verifier from the saved retrieval verification workflow."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / "tools/v330_verify_sharpness.py").read_text(encoding="utf-8")
    source = source.replace("v329_sharpness_retrieval", "v333_chemical_domain_dro")
    source = source.replace("v329", "v333").replace("v330", "v334")
    source = source.replace("v331_finalize_sharpness", "v335_finalize_domain_dro")
    source = source.replace("sharpness_retrieval", "chemical_domain_dro")
    source = source.replace("v304_learned_label_retrieval", "v315_rank_label_retrieval")
    source = source.replace("optimizer_reference()", "objective_reference()")
    source = source.replace('    fit = state["fit"]\n', '''    fit = state["fit"]
    groups = d["groups"][fit]
    domain = state["domain"]
    assert all(len(np.unique(domain[groups == group])) == 1 for group in np.unique(groups))
    np.testing.assert_allclose(state["domain_prior"], np.bincount(domain) / len(domain), rtol=0, atol=0)
    assert state["domain_centers"].shape == (c["domains"], 4096)
    assert meta["fit_only_domain_construction"]
    assert meta["domain_counts"] == np.bincount(domain).tolist()
    for history in meta["histories"]:
        for entry in history["history"]:
            np.testing.assert_allclose(sum(entry["q"]), 1., rtol=0, atol=1e-12)
            assert np.isfinite(entry["q"]).all() and min(entry["q"]) > 0
''')
    source = source.replace('        reversed_query_tolerance=1e-12)',
                            '        reversed_query_tolerance=1e-12, fit_only_domains=True, scaffold_domain_integrity=True)')
    path = ROOT / "tools/v334_verify_domain_dro.py"
    assert not path.exists()
    path.write_text(source, encoding="utf-8")
    source = (ROOT / "tools/v331_finalize_sharpness.py").read_text(encoding="utf-8")
    source = source.replace("v329_sharpness_retrieval", "v333_chemical_domain_dro").replace("v329", "v333")
    source = source.replace("v330_verify_sharpness", "v334_verify_domain_dro")
    path = ROOT / "tools/v335_finalize_domain_dro.py"
    assert not path.exists()
    path.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
