"""Detailed read-only v210 progress, including disjoint development workers."""
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'herg': 'hERG心脏毒性', 'bbb_martins': '血脑屏障', 'ames': 'AMES致突变',
    'clearance_hepatocyte_az': '肝细胞清除率', 'solubility_aqsoldb': '溶解度',
    'pgp_broccatelli': 'P-gp抑制', 'cyp2d6_substrate_carbonmangels': 'CYP2D6底物',
    'cyp3a4_veith': 'CYP3A4抑制', 'clearance_microsome_az': '肝微粒体清除率',
    'caco2_wang': 'Caco-2通透性', 'cyp3a4_substrate_carbonmangels': 'CYP3A4底物',
    'dili': '肝损伤DILI', 'vdss_lombardo': '分布容积VDSS', 'cyp2d6_veith': 'CYP2D6抑制',
    'cyp2c9_veith': 'CYP2C9抑制', 'ppbr_az': '血浆蛋白结合率', 'ld50_zhu': '急性毒性LD50',
    'lipophilicity_astrazeneca': '脂溶性', 'bioavailability_ma': '生物利用度',
    'half_life_obach': '半衰期', 'cyp2c9_substrate_carbonmangels': 'CYP2C9底物'}


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def process(pid):
    root = Path('/proc') / str(pid)
    if not root.exists():
        return dict(pid=pid, exists=False)
    stat = (root / 'stat').read_text().split()
    hz = os.sysconf('SC_CLK_TCK')
    elapsed = float(Path('/proc/uptime').read_text().split()[0]) - int(stat[21]) / hz
    cpu = (int(stat[13]) + int(stat[14])) / hz
    return dict(pid=pid, exists=True, state=stat[2], start_ticks=stat[21],
                elapsed_seconds=round(elapsed), lifetime_cpu_percent=round(100 * cpu / elapsed, 1),
                rss_gib=round(int(stat[23]) * os.sysconf('SC_PAGE_SIZE') / 1024 ** 3, 2))


def sampled_cpu(pid):
    path = Path('/proc') / str(pid) / 'stat'
    try:
        first = path.read_text().split()
        started = time.monotonic()
        time.sleep(.25)
        last = path.read_text().split()
        if first[21] != last[21]:
            return None
        ticks = int(last[13]) + int(last[14]) - int(first[13]) - int(first[14])
        return round(100 * ticks / os.sysconf('SC_CLK_TCK') / (time.monotonic() - started), 1)
    except FileNotFoundError:
        return None


def main():
    now = datetime.now(timezone.utc)
    out = ROOT / 'admet_model_outputs_v210'
    protocol = load(out / 'protocol.json')
    dev = load(out / 'development.json')
    scheduling = load(out / 'parallel_scheduling.json')
    start = load(ROOT / 'docs/v210_remote_start.json')
    results = {}
    if dev:
        for task in protocol['tasks']:
            rows = [r for r in dev['records'] if r['task'] == task]
            if rows:
                results[task] = max(rows, key=lambda r: r['objective'])
    logs = [ROOT / 'v210_train.log'] + sorted(ROOT.glob('v214_worker_*.log'))
    entries = []
    for path in logs:
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                value = json.loads(line)
                if value.get('stage') == 'train':
                    entries.append(value)
            except (ValueError, TypeError):
                pass
    timings = {(r['task'], r['part']): r['seconds'] for r in entries}
    confirmation = load(out / 'confirmation.json')
    selection = load(out / 'frozen_selection.json')
    selected = {r['task'] for r in selection['selected']} if selection else set()
    confirmations = {r['task']: r for r in confirmation['records']} if confirmation else {}
    table = []
    for task in protocol['tasks']:
        folder = out / 'models' / task
        saved = sum((folder / p / 'depth6' / f'model_{s}.cbm').exists()
                    for p in ('fold0', 'fold1') for s in (2101028, 2101029))
        partitions = [p for p in ('fold0', 'fold1') if (folder / p / 'depth6/predictions.npz').exists()]
        state = '开发模型完成，等原选择器评分' if len(partitions) == 2 else '排队'
        active_part = None
        for part in ('fold0', 'fold1'):
            p = folder / part / 'depth6'
            if p.exists() and not (p / 'predictions.npz').exists():
                active_part = part
                state = '训练' + part + '，已保存' + str(len(list(p.glob('model_*.cbm')))) + '/2种子'
                if scheduling and scheduling.get('main_temporarily_stopped') and task not in scheduling['tasks']:
                    state = '原主进程保留现场，等待并行预计算完成'
        record = results.get(task)
        if record:
            threshold = .012 if record['metric'] == 'MAE' else .008 if record['metric'] == 'Spearman' else .003
            state = '开发通过，待确认' if record['objective'] > threshold else '开发未通过'
        confirmation_models = 0
        if task in selected:
            for part in ('fold2', 'valid', 'final'):
                p = folder / part / 'depth6'
                count = len(list(p.glob('model_*.cbm')))
                confirmation_models += count
                if p.exists() and not (p / 'predictions.npz').exists():
                    active_part = part
                    state = ('最终训练' if part == 'final' else '确认训练') + part + '，已保存' + str(count) + '/2种子'
        if task in confirmations:
            state = '完整确认通过' if confirmations[task]['accepted'] else '完整确认未通过'
        table.append(dict(task=task, name=NAMES[task], saved_development_models=saved,
                          completed_partitions=len(partitions), active_part=active_part, status=state,
                          saved_confirmation_and_final_models=confirmation_models,
                          development=record, confirmation=confirmations.get(task),
                          train_seconds=sum(timings.get((task, p), 0) for p in ('fold0', 'fold1'))))
    processes = {name: process(r['pid']) for name, r in start['processes'].items()}
    if scheduling:
        processes['coordinator'] = process(scheduling['coordinator_pid'])
        processes['workers'] = [dict(**process(w['pid']), tasks=w['tasks']) for w in scheduling['workers']]
    processes['training']['sampled_cpu_percent'] = sampled_cpu(start['processes']['training']['pid'])
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        gpu = dict(status='MEASURED', utilization_percent=int(result.stdout.split(',')[0]), memory_mib=int(result.stdout.split(',')[1])) if result.returncode == 0 else dict(status='UNAVAILABLE')
    except (OSError, ValueError, subprocess.TimeoutExpired):
        gpu = dict(status='UNAVAILABLE')
    total_models = sum(r['saved_development_models'] for r in table)
    all_models_ready = sum(r['completed_partitions'] == 2 for r in table)
    cpu_quota = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    report = dict(checked_at_utc=now.isoformat(), checked_at_hong_kong=now.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),
        training_version='v210', delivery_version='v211', evaluated_development_tasks=len(results),
        completed_development_model_tasks=all_models_ready, total_tasks=21,
        saved_development_models=total_models, planned_development_models=84,
        completed_development_partitions=sum(r['completed_partitions'] for r in table),
        processes=processes, scheduling=scheduling, cpu_quota=cpu_quota,
        gpu=gpu, frozen_selection=selection,
        total_memory_gib=round(int(Path('/sys/fs/cgroup/memory.current').read_text()) / 1024 ** 3, 2),
        completed_partition_seconds=dict(median=statistics.median(timings.values()), maximum=max(timings.values())) if timings else None,
        tasks=table, failure=load(out / 'failure.json'), scheduler_failure=load(out / 'parallel_scheduling_failure.json'),
        pipeline=load(ROOT / 'admet_model_outputs_v211/pipeline_status.json'),
        confirmation=confirmation, csv_created=(ROOT / 'admet_model_outputs_v211/evidence.json').exists(),
        automatic_submission=False, official_best=51.166,
        limitation='Saved models and development gains are local evidence; no official improvement or final eligibility implied.')
    report_file = ROOT / 'docs/v210_live_progress.json'
    temp = report_file.with_name(report_file.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temp.replace(report_file)
    lines = ['# ADMET v210 训练进度', '', '更新时间（香港）：' + report['checked_at_hong_kong'], '',
        f"开发模型完成任务：{all_models_ready}/21；原选择器已评分：{len(results)}/21；已保存模型：{total_models}/84。", '',
        '当前模型：每任务2开发划分×2固定种子，CatBoost1000树depth6，CPU4线程/模型。并行仅预计算互不重叠开发任务，原选择器负责冻结选择和确认。', '',
        '| 任务 | 开发模型 | 状态 | fold0增益 | fold1增益 |', '| :-- | --: | :-- | --: | --: |']
    for row in table:
        values = {x['part']: x['gain'] for x in row['development']['parts']} if row['development'] else {}
        f = lambda part: f"{values[part]:+.6f}" if part in values else '—'
        lines.append(f"| {row['name']} | {row['saved_development_models']}/4 | {row['status']} | {f('fold0')} | {f('fold1')} |")
    lines += ['', 'MAE增益为相对误差下降；AUROC/AUPRC/Spearman为指标绝对增益。开发通过还需fold2/valid全部正增益及两份骨架bootstrap90下界>0。', '',
        '官方最好仍为v145=51.166；没有自动上传。完整验证通过后提供CSV供用户手动提交。']
    table_file = ROOT / 'docs/TRAINING_LIVE_PROGRESS.md'
    temp = table_file.with_name(table_file.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text('\n'.join(lines) + '\n')
    temp.replace(table_file)
    print(json.dumps(dict(time=report['checked_at_hong_kong'], evaluated=len(results), models_ready_tasks=all_models_ready,
        saved_models=total_models, current=[dict(task=r['task'], status=r['status']) for r in table if r['active_part']],
        scheduling=scheduling['status'] if scheduling else None, failure=report['failure'],scheduler_failure=report['scheduler_failure'])))


if __name__ == '__main__':
    main()
