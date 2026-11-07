"""Read-only distribution and duplicate-label audit; does not alter model selection."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    exp = ROOT / 'experiments/transformer_stage5_v370'
    d = dict(np.load(exp / 'official_rows.npz'))
    manifest = json.loads((exp / 'data_manifest.json').read_text(encoding='utf-8'))
    group = d['groups'][d['mi']]
    identity = d['canonical'][d['mi']]
    records = []
    for k, task in enumerate(manifest['tasks']):
        splits = {}
        for si, name in enumerate(('train', 'valid', 'test')):
            ix = np.flatnonzero((d['tid'] == k) & (d['split'] == si))
            _, counts = np.unique(group[ix], return_counts=True)
            fit = np.flatnonzero((d['tid'] == k) & (d['split'] < (2 if si == 2 else 1)))
            splits[name] = dict(rows=len(ix), unique_identities=len(np.unique(identity[ix])),
                exclusion_groups=len(counts), largest_group_rows=int(counts.max()),
                largest_group_fraction=float(counts.max()/len(ix)),
                effective_group_count=float(len(ix)**2/np.square(counts.astype(float)).sum()),
                label_quantiles=None if si == 2 else np.quantile(d['y'][ix], [0,.05,.25,.5,.75,.95,1]).tolist())
            if si:
                splits[name].update(identity_seen_in_official_fit_fraction=float(np.isin(identity[ix],identity[fit]).mean()),
                    group_seen_in_official_fit_fraction=float(np.isin(group[ix],group[fit]).mean()))
        fit = np.flatnonzero((d['tid']==k)&(d['split']<2))
        unique, reverse, count = np.unique(identity[fit], return_inverse=True, return_counts=True)
        duplicate_groups = np.flatnonzero(count > 1)
        spread = [float(np.ptp(d['y'][fit[reverse == i]])) for i in duplicate_groups]
        records.append(dict(task=task, metric=manifest['metrics'][k], splits=splits,
            repeated_identity_groups=len(duplicate_groups),
            repeated_identity_groups_with_unequal_labels=sum(v>1e-8 for v in spread),
            largest_repeated_label_range=max(spread,default=0.)))
    value = dict(status='COMPLETE', purpose='Read-only diagnostics; no retuning, test labels never accessed.',
        group_definition='Frozen historical canonical/scaffold exclusion components; counts describe structures, not independent sample size.',
        caveat='Official-split overlap does not imply model leakage. Fitting controls and pretraining explicitly exclude their own query groups.',
        records=records)
    (ROOT/'docs/v370_data_diagnostics.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    lines=['# 数据诊断（不参与本轮选型）','','官方标签仅使用train/valid；test只查看结构和行数。组数和有效组数是结构集中度诊断，不是独立样本数量估计。',
        '下表的valid来自官方原划分；训练控制另外排除查询组，所以官方原划分的结构重合不能直接解释为模型泄漏。','',
        '|任务|train/valid/test|valid最大组占比|test最大组占比|train+valid重复结构组|其中标签不一致|',
        '|---|---|---:|---:|---:|---:|']
    for r in records:
        s=r['splits']
        lines.append(f"|{r['task']}|{s['train']['rows']}/{s['valid']['rows']}/{s['test']['rows']}|{s['valid']['largest_group_fraction']:.1%}|{s['test']['largest_group_fraction']:.1%}|{r['repeated_identity_groups']}|{r['repeated_identity_groups_with_unequal_labels']}|")
    lines += ['','结构重复后的标签差异可能来自测量条件、实验噪声、非立体canonical合并等，需要逐项核对；不能直接删成“错误标签”。',
        'AMES valid集中于一个苯环骨架组；LD50 fold2只有9个冻结组。单一分区的改进不足以支持隐藏集大幅收益。',
        '建议下一阶段先重建完整模型的诚实OOF，明确任务单位和测量条件，再决定是否值得继续增加网络复杂度。本轮仍使用已冻结的模型与融合规则。','']
    (ROOT/'docs/V370_DATA_DIAGNOSTICS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps([dict(task=r['task'],valid_largest=r['splits']['valid']['largest_group_fraction'],duplicate_conflicts=r['repeated_identity_groups_with_unequal_labels']) for r in records]))


if __name__ == '__main__':
    main()
