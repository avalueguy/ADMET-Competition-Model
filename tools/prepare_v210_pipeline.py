"""Prepare reviewed glue for the new v210 representation; no training here."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    translations = {
        'v207': 'v210', 'v208': 'v211', 'v209': 'v212',
        'denoising_encoder': 'standardized_chemistry',
        'denoising_batch': 'standardized_batch',
    }
    for source, target in (
        ('v208_verify_denoising_encoder.py', 'v211_verify_standardized_chemistry.py'),
        ('v209_package_denoising_batch.py', 'v212_package_standardized_batch.py'),
        ('finalize_v208_local.py', 'finalize_v211_local.py'),
    ):
        text = (ROOT / 'tools' / source).read_text(encoding='utf-8-sig')
        for old, new in translations.items():
            text = text.replace(old, new)
        text = text.replace('Independent denoising encoder reload', 'Independent standardized representation and model reload')
        text = text.replace('model.configure()\n', '')
        text = text.replace('    model.verify_pretraining(d)', '    model.verify_features(d)')
        # Removing configure must retain the original indentation of the load.
        text = text.replace('        d = v.load()', '    d = v.load()')
        text = text.replace('denoising encoder fusion', 'standardized chemistry fusion')
        text = text.replace('distinct v210 denoising encoder', 'distinct v210 standardized chemistry')
        text = text.replace('failed denoising batch', 'failed standardized chemistry batch')
        if source == 'finalize_v208_local.py':
            start = text.index("    pretraining = load(")
            end = text.index("    assert confirmation['status']", start)
            text = text[:start] + """    feature_audit = load(ROOT / 'admet_model_outputs_v210/features/audit.json')
    feature_check = load(ROOT / 'admet_model_outputs_v210/features/independent_verification.json')
    assert feature_check['status'] == 'PASS' and feature_check['all_structures_regenerated']
    assert feature_check['feature_sha256'] == feature_audit['sha256']
    assert sha(ROOT / 'admet_model_outputs_v210/features/normalized.npz') == feature_audit['sha256']
    assert sha(ROOT / 'admet_model_outputs_v210/features/representations.joblib') == feature_check['representations_sha256']
""" + text[end:]
            text = text.replace("state['releases']['201']", "state['releases']['211']")
        assert 'v207' not in text and 'v208' not in text
        ast.parse(text)
        path = ROOT / 'tools' / target
        assert not path.exists(), path
        path.write_text(text, encoding='utf-8')
    source = ROOT / 'tools/v210_standardized_chemistry.py'
    runner = ROOT / 'tools/v148_experiment_runner.py'
    plan = dict(version='v210', delivery_version='v211', baseline='v145', official_baseline_score=51.166,
        created_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(source), runner_sha256=sha(runner),
        model='Neutral organic-parent and bounded canonical-tautomer representations plus original augmented chemistry, two-seed CatBoost',
        distinct_from_completed_routes='v95/auditv101 identities and v96 neutral assay matching did not train these two normalized molecular fingerprint/descriptor representations. No prior route or model rerun.',
        representations=dict(normalization='Largest organic fragment, uncharge; then RDKit bounded canonical tautomer',
            enumeration=dict(max_tautomers=256, max_transforms=1024, status_flag=True),
            per_representation='2048 radius2 chiral countMorgan +1024 radius3 featureMorgan +214 descriptors',
            original_valence_exceptions='Retain original graph in both views and explicit flag, no silent repair',
            unanticipated_normalization_error='Fail closed; no silent zero/fallback',
            source='Official SMILES only, per-molecule transform; no labels, query/fit adaptation or learned preprocessing'),
        tasks='All21 improvable endpoints ordered by original official score; HIA protected',
        configs=[dict(depth=6, trees=1000)], seeds=[2101028,2101029], weights=[.2,.45],
        optimizer='CatBoost CPU4threads,lr.03,l2=30,subsample.85,border64. Logloss/MAE/ranknormalRMSE; no early stopping',
        selection='Unchangedv148 maxminimumfold0/1; freeze single choice beforefold2/valid; no fallback',
        confirmation='All4 point gains>0, original practical mean threshold, both scaffold-bootstrap90 lower bounds>0',
        tests_before_training=['Source row mapping and all-label perturbation', 'Known salt and keto/enol invariance',
            'Per-molecule deterministic order', 'Actual classification/MAE/Spearman estimator independent save/reload', 'Query label isolation'],
        verification='Regenerate every normalized structure/feature; independently reload all accepted models; original gates/fusion/template bytes/backup hashes',
        source_immutable_after_start=True, test_labels_used=False, new_external_data=False,
        pretrained_weights_downloaded=False, official_gain_guaranteed=False,
        limitations='Normalized representations may omit meaningful salt/charge/tautomer information; originals remain available. Public labels reused and historical upstream OOF approximate; no official gain estimate.')
    save(ROOT / 'docs/v210_training_plan.json', plan)
    save(ROOT / 'docs/v210_source_review.json', dict(status='STATIC_REVIEW_PASS_SMOKE_REQUIRED',
        frozen_source_sha256=sha(source), smoke_required=True,
        checks=['Per-molecule originalSMILES matched to official structure signature; no labels consulted',
                'Original valence exceptions explicit; unexpected normalization failures stop experiment',
                'Two seeds one fixed configuration, no expanding search using confirmation labels',
                'Originalv148 unchanged; independent all-feature regeneration and model reload required',
                'New local finalizer checks correctv211 release key and normalized feature hashes']))
    print(json.dumps(dict(status='PREPARED', source_sha256=sha(source))))


if __name__ == '__main__':
    main()
