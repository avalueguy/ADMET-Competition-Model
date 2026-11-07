"""Prepare a distinct observation-noise protocol before any development fit."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'tools/v264_replicate_noise_gp.py'
    model = source.read_text(encoding='utf-8')
    assert 'CONFIGS = (dict(prior_df=5., max_iterations=60),)' in model
    diagnostic = json.loads((ROOT / 'docs/v263_fit_label_reliability.json').read_text())
    focus = [r for r in diagnostic['records'] if r['task'] in ('ppbr_az', 'clearance_hepatocyte_az')]
    review = dict(status='FIT_ONLY_MECHANISM_REVIEW_COMPLETE', at_utc=datetime.now(timezone.utc).isoformat(),
        diagnostic='docs/v263_fit_label_reliability.json', diagnostic_sha256=sha(ROOT / 'docs/v263_fit_label_reliability.json'),
        focus=[dict(task=r['task'], parts=[dict(part=p['part'], fit_rows=p['fit_rows'],
                    exact_structure=p['exact_structure']) for p in r['parts']]) for r in focus],
        labels_changed=False, confirmation_scores_used=False, new_predictions=False,
        historical_overlap=[
            dict(path='admet_v61_run.py', mechanism='inverse identity count sample weighting'),
            dict(path='admet_v68_kernel.py', mechanism='Tanimoto SVR with inverse identity count risk weighting'),
            dict(path='admet_v71_v56_catboost.py', mechanism='CatBoost inverse identity count risk weighting'),
            dict(path='admet_v78_median.py', mechanism='XGBoost MAE/Huber with inverse canonical count risk weighting'),
            dict(path='admet_multitask.py', mechanism='canonical mean aggregation, with equal certainty'),
            dict(path='tools/v92_full_kernel.py', mechanism='homoscedastic fixed-ridge Tanimoto models'),
            dict(path='tools/v244_conditional_density.py', mechanism='per-row neural logistic mixture, no replicate-calibrated observation covariance')],
        rejected_interpretations=['Do not treat rounded clearance boundaries as established censoring',
            'Do not merge salts/stereochemistry or alter labels',
            'Do not claim conflicts prove measurement error; unrecorded assay conditions can differ',
            'Inverse-frequency weighting and simple mean aggregation alone repeat prior work'],
        decision='Test one bounded fit-only replicate-variance observation model on PPBR and hepatocyte, with existing kernels. No query-based variance tuning, no sweep or failed-confirmation fallback.',
        limitation='All public labels have historical reuse; neither diagnostic nor a fitted model proves hidden-test gain.')
    save(ROOT / 'docs/v263_review_decision.json', review)
    plan = dict(version='v264', delivery_version='v265', baseline='v145', official_baseline_score=51.166,
        created_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(source),
        runner_sha256=sha(ROOT / 'tools/v148_experiment_runner.py'),
        tasks=['ppbr_az', 'clearance_hepatocyte_az'], protected_tasks=20,
        development_partitions=4, deterministic_models=4, seeds=[],
        configs=[dict(prior_df=5., max_iterations=60)], weights=[.2, .45],
        hypothesis='Conflicting exact-original-structure observations are abundant in both training folds for these two endpoints. A Gaussian observation model can separate replicate scatter from molecular signal instead of giving every aggregated mean the same certainty or using inverse-count risk weights.',
        mechanism='Exact original-SMILES identities within current fit rows only. PPBR raw target and log1p clearance; median/IQR fit standardization. Group sample mean and within-group scatter are Gaussian sufficient statistics. Variance shrinks toward pooled within-group variance with 5 prior degrees, then divides by count. A bounded fit marginal-likelihood optimizer calibrates signal amplitude and additional group-level variance. Gaussian posterior median; raw PPBR clipped 0..100, clearance expm1. No raw label edited or deleted.',
        inputs='Existing equal six-view Tanimoto .75 plus fit-only physical RBF .15 and target-excluded other-assay context RBF .10. Kernels deliberately reuse established features; observation covariance is the proposed distinction.',
        historical_review='docs/v263_review_decision.json',
        optimization='One deterministic start log([1,.05]); L-BFGS-B log signal[-5,3], log extra group variance[-9,2], max60, ftol1e-10, gtol1e-5. Optimizer failure is a failure, never a candidate or alternate initialization retry.',
        scheduling='One CPU4thread worker, two tasks and four development partitions; no GPU. All worker exit0 before unchanged selector. 19 identity controls within runner21; HIA excluded.',
        preconditions='Analytic replicate variance reference, marginal likelihood reference and finite-difference gradient, PSD, query label perturbation, actual optimizer save/reload and query order smoke; original frozen runner/baseline sig/105 caches/dependencies before launch.',
        selection='Original v148 one config and .2/.45 blend max-min on fold0/1, frozen before fold2/valid. All4 gains>0, original mean thresholds, two scaffold-bootstrap90 lower bounds>0, no failure fallback.',
        accepted_verification='Independent full feature/variance reconstruction, model reload, original gate/fusion/hash/query order, original Drug_ID strings and nonprediction bytes, full archive and dependency restore verification before manual delivery.',
        source_immutable_after_start=True, test_labels_used=False, new_external_data=False,
        manual_submission_only=True, official_gain_guaranteed=False,
        limitations='Gaussian latent response, variance shrinkage and reused kernels may be misspecified. Conflicts can reflect unrecorded assay conditions. Historical proxy controls are not complete exact v145 OOF. Public labels reused.')
    save(ROOT / 'docs/v264_training_plan.json', plan)
    save(ROOT / 'docs/v264_source_review.json', dict(status='REVIEWED_BEFORE_DEVELOPMENT',
        review='docs/v263_review_decision.json', label_isolation='Only fit labels enter sufficient statistics/optimizer. Other-assay context uses unchanged target-excluded loader.',
        novelty='Replicate-derived heterogeneous observation covariance with fit-only marginal-likelihood calibration. No claim that the existing Tanimoto/RBF inputs are new.',
        prior_failed_routes_retried=False, parameters_chosen_without_query_scores=True,
        no_confirmation_fallback=True, thresholds_unchanged=True))

    # Reuse audited assembly code, replacing only route-specific feature checks.
    verify = (ROOT / 'tools/v257_verify_exposed_polarity.py').read_text(encoding='utf-8')
    verify = verify.replace('v256_exposed_polarity', 'v264_replicate_noise_gp').replace('v257_exposed_polarity', 'v265_replicate_noise_gp').replace('v257', 'v265').replace('v256', 'v264')
    start = verify.index("    feature_audit = __import__")
    end = verify.index('    original = table', start)
    verify = verify[:start] + verify[end:]
    verify = verify.replace('surface_feature_audit=feature_audit,', '').replace('surface_reconstruction=feature_reconstruction,', '')
    verify = verify.replace('Independent exposed polarity', 'Independent replicate noise')
    (ROOT / 'tools/v265_verify_replicate_noise.py').write_text(verify, encoding='utf-8')
    final = (ROOT / 'tools/finalize_v257_local.py').read_text(encoding='utf-8')
    final = final.replace('v256_exposed_polarity', 'v264_replicate_noise_gp').replace('v257_exposed_polarity', 'v265_replicate_noise_gp')
    final = final.replace('v257', 'v265').replace('v256', 'v264').replace('conformer_exposed_polarity', 'replicate_noise_gaussian_process')
    final = final.replace("state['releases']['257']", "state['releases']['265']").replace('exposed polarity', 'replicate noise')
    start = final.index('    feature_audit = load(')
    end = final.index('    checks = report', start)
    final = final[:start] + final[end:]
    (ROOT / 'tools/finalize_v265_local.py').write_text(final, encoding='utf-8')
    package = (ROOT / 'tools/v259_package_surface_batch.py').read_text(encoding='utf-8')
    package = package.replace('v257_exposed_polarity', 'v265_replicate_noise_gp').replace('v256_exposed_polarity', 'v264_replicate_noise_gp')
    package = package.replace('v257_verify_exposed_polarity', 'v265_verify_replicate_noise')
    package = package.replace('v260_surface_driver', 'v267_replicate_driver').replace('v261_start_surface', 'v269_start_replicate')
    package = package.replace('v257', 'v265').replace('v256', 'v264').replace('v260_worker', 'v267_worker').replace('exposed polarity', 'replicate noise')
    for line in ["                  ROOT / 'docs/v255_surface_feasibility.json',\n",
                 "                  ROOT / 'tools/v258_build_surface.py',\n",
                 "                  ROOT / 'v264_feature_build.log',\n"]:
        package = package.replace(line, '')
    package = package.replace("                  ROOT / 'tools/v269_start_replicate.py',", "                  ROOT / 'tools/v269_start_replicate.py', ROOT / 'tools/v268_prepare_replicate_protocol.py',\n                  ROOT / 'docs/v263_fit_label_reliability.json', ROOT / 'docs/v263_review_decision.json',")
    (ROOT / 'tools/v266_package_replicate.py').write_text(package, encoding='utf-8')
    driver = (ROOT / 'tools/v260_surface_driver.py').read_text(encoding='utf-8')
    driver = driver.replace('v256_exposed_polarity', 'v264_replicate_noise_gp').replace('v256', 'v264').replace('v260', 'v267')
    (ROOT / 'tools/v267_replicate_driver.py').write_text(driver, encoding='utf-8')
    launch = (ROOT / 'tools/v261_start_surface.py').read_text(encoding='utf-8')
    launch = launch.replace('v256_exposed_polarity', 'v264_replicate_noise_gp').replace('v260_surface_driver', 'v267_replicate_driver')
    launch = launch.replace('v259_package_surface_batch', 'v266_package_replicate').replace('v259_pipeline', 'v266_pipeline')
    launch = launch.replace('v256', 'v264')
    (ROOT / 'tools/v269_start_replicate.py').write_text(launch, encoding='utf-8')
    files = [source, ROOT / 'tools/v265_verify_replicate_noise.py', ROOT / 'tools/finalize_v265_local.py',
             ROOT / 'tools/v266_package_replicate.py', ROOT / 'tools/v267_replicate_driver.py',
             ROOT / 'tools/v268_prepare_replicate_protocol.py', ROOT / 'tools/v269_start_replicate.py',
             ROOT / 'docs/v264_training_plan.json', ROOT / 'docs/v264_source_review.json',
             ROOT / 'docs/v263_review_decision.json', ROOT / 'tools/v148_experiment_runner.py']
    save(ROOT / 'docs/v264_frozen_sources.json', dict(status='FROZEN_BEFORE_SMOKE_AND_DEVELOPMENT',
        files=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in files],
        credentials_stored=False, source_immutable_after_start=True))
    print(json.dumps(dict(status='PREPARED', source_sha256=sha(source), files=len(files))))


if __name__ == '__main__':
    main()
