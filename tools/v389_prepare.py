"""Freeze the restricted two-expert coefficient search before evaluating it."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    exp=ROOT/'experiments/regression9_v389';exp.mkdir(exist_ok=True);assert not(exp/'plan.json').exists()
    dev={v:json.loads((ROOT/f'experiments/regression9_{v}/outputs/development.json').read_text(encoding='utf-8'))for v in('v386','v387')}
    fixed={r['task']:{v:next(x['best']['config']for x in d['records']if x['task']==r['task'])for v,d in dev.items()}for r in dev['v386']['records']}
    plan=dict(version='v389',registered_at_utc=datetime.now(timezone.utc).isoformat(),baseline='v387:51.314',
        baseline_csv_sha256=json.loads((ROOT/'docs/current_best_after_v387_20261106.json').read_text())['csv_sha256'],
        source_sha256=sha(ROOT/'tools/v389_combination.py'),fixed_experts=fixed,grid=[0,.05,.1,.2,.35,.5,.75],maximum_total_new_weight=.75,
        formula='(1-a-b)*complete_v387_control+a*frozen_v386_expert+b*frozen_v387_expert',
        selection='Unique max-min relative MAE improvement fold0/1; rank tasks also positive Spearman in both; ties lower total weight then lower a.',
        confirmation='Frozen choice fold2/valid; all4 MAE and all4 Spearman positive; no fallback; strict1.2percent developer magnitude and bootstrap positive.',
        limit='Past public partitions and frozen component configurations already researched. This is not independent blind validation; no hidden labels, no per-molecule routing.',
        fitting_location='Authorized25870 remote only; any missing predictions fit frozen original models there; no local fitting.')
    (exp/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'docs/V389_COMBINATION_PLAN.md').write_text('# v389：两类固定专家的受限融合\n\n以官方51.314关联v387为完整基线。两版只在脂溶性有差异，因此整任务取优等于v387，不能声称直接合并获得额外收益。\n\n固定v386/v387先前各任务开发所选配置，仅搜索非负两系数，总新增权重不超过75%。全部系数组合完成后由fold0/1 max-min选择唯一配方，冻结后fold2/valid确认，失败不换候选。所有拟合和系数选择仅指定远端。公开分区曾复用，不称新盲测。\n',encoding='utf-8')
    print(json.dumps(dict(source_sha256=plan['source_sha256'],fixed_tasks=len(fixed),plan_sha256=sha(exp/'plan.json'))))
if __name__=='__main__':main()
