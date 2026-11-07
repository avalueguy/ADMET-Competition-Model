"""Write a separate terminal report including the direct-AP experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "tools/v339_write_optimization_review.py"
    source = path.read_text(encoding="utf-8")
    old = '((329, 330, "SAM 抑制过拟合"), (333, 334, "化学结构域稳健训练"))'
    new = '((329, 330, "SAM 抑制过拟合"), (333, 334, "化学结构域稳健训练"), (340, 341, "直接优化 AUPRC"))'
    assert source.count(old) == 1
    source = source.replace(old, new)
    source = source.replace('docs/v339_optimization_review.json', 'docs/v345_optimization_review.json')
    source = source.replace('OPTIMIZATION_V329_V333_20261101.md', 'OPTIMIZATION_THROUGH_V340_20261101.md')
    source = source.replace('# v329 / v333 优化与核验结论', '# v329 / v333 / v340 优化与核验结论')
    source = source.replace('- 两轮均在评分前锁定源码与配置；',
        '- v340：对五个 AUPRC 任务直接优化可微平均精度与 BCE 的等权组合；固定温度 0.2，均匀 256 行批次，沿用原网络和种子，从头训练。损失已用独立标量公式、有限差分梯度和离散 AP 极限核对。\n- 三轮均在评分前锁定源码与配置；')
    source = source.replace('v339_optimization_review.json)', 'v345_optimization_review.json)')
    source = source.replace('[计分差距](v338_current_objective_audit.json)',
        '[v340 终态](v340_final_status.json)、[v341 核验](v341_verification.json)、[计分差距](v338_current_objective_audit.json)')
    exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})


if __name__ == "__main__":
    main()
