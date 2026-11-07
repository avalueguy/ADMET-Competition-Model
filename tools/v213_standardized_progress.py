"""Read-only progress snapshot for the frozen v210 experiment."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    start = load(ROOT / 'docs/v210_remote_start.json')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    processes = {}
    for name, expected in start['processes'].items():
        folder = Path('/proc') / str(expected['pid'])
        if not folder.exists():
            processes[name] = dict(pid=expected['pid'], exists=False)
            continue
        stat = (folder / 'stat').read_text().split()
        cmd = (folder / 'cmdline').read_bytes().replace(bytes([0]), b' ').decode()
        assert start['boot_id'] == boot and stat[21] == expected['start_ticks']
        assert expected['script'] in cmd
        processes[name] = dict(pid=expected['pid'], exists=True, state=stat[2],
                               start_ticks=stat[21], cmdline=cmd)
    out = ROOT / 'admet_model_outputs_v210'
    dev = load(out / 'development.json')
    tasks = []
    if dev:
        for task in dict.fromkeys(r['task'] for r in dev['records']):
            candidates = [r for r in dev['records'] if r['task'] == task]
            best = max(candidates, key=lambda r: r['objective'])
            threshold = .012 if best['metric'] == 'MAE' else .008 if best['metric'] == 'Spearman' else .003
            tasks.append(dict(task=task, metric=best['metric'], objective=best['objective'],
                              weight=best['weight'], development_threshold=threshold,
                              passes_development=best['objective'] > threshold,
                              gains={p['part']: p['gain'] for p in best['parts']}))
    features = load(out / 'features/audit.json')
    plan = load(ROOT / 'docs/v210_training_plan.json')
    source = hashlib.sha256((ROOT / 'tools/v210_standardized_chemistry.py').read_bytes()).hexdigest()
    assert source == plan['source_sha256']
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), boot_id=boot,
        training_version='v210', delivery_version='v211', processes=processes,
        source_sha256=source, source_matches_frozen_plan=True,
        development_status=dev['status'] if dev else 'NO_TASK_COMPLETE',
        completed_development_tasks=len(tasks), development=tasks,
        features=dict(shape=features['shape'], sha256=features['sha256'],
            original_graph_exceptions=len(features['original_graph_exceptions']),
            bounded_enumeration_rows=len(features['bounded_enumeration_rows']),
            descriptor_exception_rows=len(features['descriptor_exception_rows'])) if features else None,
        failure=load(out / 'failure.json'), confirmation=load(out / 'confirmation.json'),
        pipeline=load(ROOT / 'admet_model_outputs_v211/pipeline_status.json'),
        no_candidate=load(ROOT / 'admet_model_outputs_v211/no_candidate.json'),
        evidence_exists=(ROOT / 'admet_model_outputs_v211/evidence.json').exists(),
        archive_exists=(ROOT / 'releases/v211_standardized_chemistry_20261028.tar.gz').exists(),
        log_tail=(ROOT / 'v210_train.log').read_text()[-3500:])
    (ROOT / 'docs/v210_progress_latest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
