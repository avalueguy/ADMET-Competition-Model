"""Preserve the v284 implementation defect and freeze a single schema correction.

This creates new files only. It does not alter or rerun any frozen experiment,
change model parameters, inspect new scores, or start training.
"""
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = ROOT / path
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    now = datetime.now(timezone.utc).isoformat()
    assert sha(ROOT / 'tools/v284_stereo_context_repair.py') == '3f7f111a75053710369247e89cf835fa2567fceaf8e54a7c2a3187350582df91'
    assert sha(ROOT / 'tools/v148_experiment_runner.py') == '8f6d93f517d1a4acf6877f3effa63e3b2c171141b5a637b8ad51573498c0b871'
    development = read('docs/v284_development.json')
    best = [max((r for r in development['records'] if r['task'] == t), key=lambda r:r['objective'])
            for t in ('herg', 'cyp3a4_substrate_carbonmangels')]
    result = dict(status='COMPLETE_NO_CSV_IMPLEMENTATION_DEFECT_FOUND', training_version='v284',
        delivery_version='v285', baseline='v145', official_best=51.166, at_utc=now,
        development_best=best, confirmation=[], accepted_tasks=[], new_csv_created=False,
        new_upload_performed=False, archive=read('docs/v285_local_archive_verification.json'),
        exited_pids=[3026,3027,3034,3276], source_unchanged=True,
        interpretation='Completed normally but fails intended input-schema contract: replaced 42 of 105 context columns, leaving 63 old columns. Raw scores belong to the defective 603-column implementation; not evidence rejecting the intended 540-column replacement. Preserve immutable source/archive. No score-based parameter change.')
    save('docs/v285_final_status.json', result)
    save('docs/v284_followup.json', dict(**result, model_training_active=False,
        source_sha256=sha(ROOT/'tools/v284_stereo_context_repair.py'),
        signature=development['signature'], final_status='docs/v285_final_status.json',
        next_action='New version may repair the proven 63-column retention bug only, keeping originally fixed task/model/seeds/weights/gates. Do not restart old PIDs.'))
    defect = dict(status='CONFIRMED_INPUT_SCHEMA_DEFECT', at_utc=now,
        source='tools/v284_stereo_context_repair.py', source_sha256=sha(ROOT/'tools/v284_stereo_context_repair.py'),
        source_line='width = 2*len(cols)', actual_original_context_columns=105,
        original_physchem_columns=435, expected_total_columns=540, actual_total_columns=603,
        retained_legacy_context_columns=63,
        dependency_evidence=['tools/v125_low_task_baseline.py: features uses assays.assays',
            'tools/v96_normalized_assays.py: assays returns exactMedian/exactMask/neutralMedian/neutralMask/neutralRange (5*21 columns)',
            'Remote original.features and v284.design observed 305x540 versus 305x603 for hERG/fold0'],
        repair='Replace all 5*n context columns, verify exact original context prefix and physical suffix, preserve total width. Add independent schema regression guards.',
        old_source_and_archive_preserved=True, model_parameters_changed=False,
        original_seeds_changed=False, gates_changed=False, new_scores_used=False,
        original_review_sha_note='v284_source_review references an earlier prefreeze source SHA; actual freeze/plan/archive agree on 3f7f111a. Preserve mismatch as historical evidence; new source review must exactly match final frozen SHA.')
    save('docs/v289_schema_defect_review.json', defect)

    replacements = [
        ('v288_start_stereo_repair','v293_start_stereo_schema_fix'),
        ('v287_stereo_driver','v292_stereo_schema_driver'),
        ('v286_package_stereo_repair','v291_package_stereo_schema_fix'),
        ('v285_verify_stereo_repair','v290_verify_stereo_schema_fix'),
        ('v284_stereo_context_repair','v289_stereo_schema_fix'),
        ('finalize_v285_local','finalize_v290_local'),
        ('v145_stereo_specific_context','v145_stereo_specific_context_fixed_schema'),
        ('v285_stereo_context_repair_20261030','v290_stereo_schema_fix_20261031'),
        ('v284','v289'), ('v285','v290'), ('v286','v291'), ('v287','v292'), ('v288','v293'),
        ('stereo_context_repair','stereo_schema_fix'), ('20261030','20261031')]
    files = [
        ('tools/v284_stereo_context_repair.py','tools/v289_stereo_schema_fix.py'),
        ('tools/v285_verify_stereo_repair.py','tools/v290_verify_stereo_schema_fix.py'),
        ('tools/v286_package_stereo_repair.py','tools/v291_package_stereo_schema_fix.py'),
        ('tools/v287_stereo_driver.py','tools/v292_stereo_schema_driver.py'),
        ('tools/v288_start_stereo_repair.py','tools/v293_start_stereo_schema_fix.py'),
        ('tools/finalize_v285_local.py','tools/finalize_v290_local.py')]
    for old, new in files:
        source = (ROOT/old).read_text(encoding='utf-8-sig')
        for a,z in replacements:
            source=source.replace(a,z)
        if new.endswith('v289_stereo_schema_fix.py'):
            start=source.index('    # original.v.features starts')
            stop=source.index('\n\n\ndef original_replay',start)
            source=source[:start]+'''    old, old_cols, old_source, _, _ = v.old.old.old.assays.assays(d, k, part)
    np.testing.assert_array_equal(cols, old_cols)
    np.testing.assert_array_equal(source, old_source)
    width = old.shape[1]
    assert width == 5*len(cols) == new.shape[1] == 105
    raw = d['views']['physchem']
    assert raw.shape[1] == 435 and x.shape[1] == qx.shape[1] == width+raw.shape[1]
    # Check actual producer layout, not a guessed number of context blocks.
    np.testing.assert_array_equal(x[:, :width], old[fit])
    np.testing.assert_array_equal(qx[:, :width], old[query])
    for rows, values in ((fit, x), (query, qx)):
        physical = raw[d['mi'][rows]].astype(np.float32).copy()
        physical[~np.isfinite(physical)] = np.nan
        np.testing.assert_array_equal(values[:, width:], physical)
    xx = np.c_[new[fit], x[:, width:]].astype(np.float32)
    zz = np.c_[new[query], qx[:, width:]].astype(np.float32)
    assert xx.shape == x.shape and zz.shape == qx.shape
    np.testing.assert_array_equal(xx[:, :width], new[fit])
    np.testing.assert_array_equal(zz[:, :width], new[query])
    np.testing.assert_array_equal(xx[:, width:], x[:, width:])
    np.testing.assert_array_equal(zz[:, width:], qx[:, width:])
    # Identity replacement with original context must reproduce original inputs.
    np.testing.assert_array_equal(np.c_[old[fit], x[:, width:]], x)
    np.testing.assert_array_equal(np.c_[old[query], qx[:, width:]], qx)
    return dict(fit=fit, query=query, source=source, cols=cols, x=xx, qx=zz)'''+source[stop:]
            source=source.replace("'Fit-only audit supports hERG/CYP3A4 substrate; CYP2D6 excluded by declared '",
                "'Corrects proven v284 63-column legacy-context retention defect only; original 540-column layout enforced. '")
            source=source.replace("checks.append(dict(task=task, actual_cuda_training=True, save_reload=True, row_order=True))",
                "checks.append(dict(task=task, actual_cuda_training=True, save_reload=True, row_order=True, original_input_width=540, replacement_input_width=a['x'].shape[1], context_blocks=5, legacy_columns_retained=0))")
            source=source.replace("target_nonfit_label_isolation=True, enantiomer_separation=True,",
                "target_nonfit_label_isolation=True, enantiomer_separation=True, complete_context_replacement=True, original_physical_suffix_exact=True, identity_replacement_reference=True,")
        if new.endswith('v291_package_stereo_schema_fix.py'):
            source=source.replace("ROOT / 'docs/v289_original_models.json'])",
                "ROOT / 'docs/v289_original_models.json', ROOT / 'docs/v289_schema_defect_review.json',\n                  ROOT / 'docs/v285_final_status.json', ROOT / 'tools/prepare_v289_schema_fix.py'])")
            source=source.replace("ROOT / 'docs/v289_frozen_sources.json',",
                "ROOT / 'docs/v289_frozen_sources.json', ROOT / 'docs/v289_remote_start.json',")
        target=ROOT/new
        assert not target.exists(),target
        target.write_text(source,encoding='utf-8')

    save('docs/v289_original_models.json',read('docs/v284_original_models.json'))
    plan=read('docs/v284_training_plan.json')
    plan.update(version='v289',delivery_version='v290',created_at_utc=now,
        source_sha256=sha(ROOT/'tools/v289_stereo_schema_fix.py'),
        implementation_defect='docs/v289_schema_defect_review.json',
        correction='Complete 105-column replacement; original 435 physical columns unchanged.540 columns total. No parameter/seed/task/weight change. No new architecture.',
        preconditions='Original all5part component replay, exact producer/prefix/suffix/width identity regression guards, enantiomer and atom order identities, target-nonfit-label isolation, actualCUDA checkpoint reload and query-order smoke,105controls/dependency hashes.')
    save('docs/v289_training_plan.json',plan)
    save('docs/v289_source_review.json',dict(status='PREDECLARED_IMPLEMENTATION_CORRECTION',
        at_utc=now,source_sha256=plan['source_sha256'],review='docs/v289_schema_defect_review.json',
        scopes=['Only context prefix boundary changes from42 to105; no legacy context retained',
                'Original model, seeds, folds, labels, blend/gates preserved',
                'No failed v284 model or prediction reused', 'Old source, log, checkpoints and archive untouched'],
        known_score_reuse=True,fit_predictions_called_oof=False,hidden_gain_guaranteed=False))
    paths=[new for _,new in files]+['tools/prepare_v289_schema_fix.py',
        'docs/v289_schema_defect_review.json','docs/v289_training_plan.json',
        'docs/v289_source_review.json','docs/v289_original_models.json',
        'tools/v148_experiment_runner.py','docs/v283_review_decision.json',
        'docs/v283_stereochemical_context_audit.json']
    frozen=dict(status='FROZEN_BEFORE_SMOKE_AND_TRAINING',created_at_utc=now,
        files=[dict(path=p,sha256=sha(ROOT/p)) for p in paths])
    save('docs/v289_frozen_sources.json',frozen)
    target=ROOT/'releases/v289_frozen_sources_20261031.tar.gz'
    assert not target.exists()
    with tarfile.open(target,'w:gz') as tf:
        for p in paths+['docs/v289_frozen_sources.json']:
            tf.add(ROOT/p,arcname=p)
    state=read('CURRENT_STATUS.json')
    state.update(current_followup='docs/v289_followup.json', active_experiment='v289',
        active_experiment_status='FROZEN_SCHEMA_CORRECTION_NOT_STARTED',
        current_state_checked_at_utc=now,running_new_training_processes=0,
        running_new_training_processes_note='v284/v285 ended; proven input schema defect preserved. v289 frozen implementation correction, not yet started.',
        new_submission_created=False,recommended_submission_version=None,
        priority_next_step='Run frozen v289 schema regression and actual label-isolation/reload smoke, verify105controls/dependencies then start once. No scores before protocol freeze.')
    state['v285_result']=result
    state['active_iteration']=dict(version='v289',delivery_version='v290',baseline='v145',status='FROZEN_SCHEMA_CORRECTION_NOT_STARTED',running_training_processes=0)
    (ROOT/'CURRENT_STATUS.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save('docs/v289_followup.json',dict(status='FROZEN_SCHEMA_CORRECTION_NOT_STARTED',
        training_version='v289',delivery_version='v290',baseline='v145',official_best=51.166,
        source_sha256=plan['source_sha256'],training_plan='docs/v289_training_plan.json',
        defect_review='docs/v289_schema_defect_review.json',frozen_sources='docs/v289_frozen_sources.json',
        frozen_archive=target.relative_to(ROOT).as_posix(),frozen_archive_sha256=sha(target),
        tasks=plan['tasks'],host='connect.nmb2.seetacloud.com',port=24365,
        remote_workspace='/root/autodl-tmp/v76',model_training_started=False,
        source_immutable_after_start=True,credentials_stored=False,automatic_platform_submission=False,
        pipeline_status='admet_model_outputs_v290/pipeline_status.json',
        scheduler_status='admet_model_outputs_v289/parallel_scheduling.json',
        remote_start='docs/v289_remote_start.json',log='v289_stereo_schema_fix.log',
        worker_logs='v292_worker_*.log',archive_prefix='v290_stereo_schema_fix_20261031',
        local_finalizer='tools/finalize_v290_local.py',previous_result='docs/v285_final_status.json',
        next_steps=['Frozen smoke then105cache/dependency/alloriginalreplay preflight before start',
                    'No failedmodel reuse or parameter change, original gates only',
                    'Download archive/manifest/info, restore allmembers/dependencies; acceptedonly finalizer/manualCSV']))
    print(json.dumps(dict(status='FROZEN_NOT_STARTED',source_sha256=plan['source_sha256'],files=[new for _,new in files],frozen_archive_sha256=sha(target))),flush=True)


if __name__ == '__main__':
    main()
