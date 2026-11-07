"""Create no-fit evidence receivers without changing frozen training sources."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def common(source, version, route):
    old_version = 'v385' if version == 'v386' else 'v384'
    s = source.replace(old_version, version).replace(old_version.upper(), version.upper())
    s = s.replace('v382', 'v385').replace('51.299', '51.308')
    s = s.replace("base['csv_sha256']", "inputs['baseline_csv_sha256']")
    s = s.replace("base=inputs['baseline'];", "base=inputs['baseline'];assert inputs['baseline_csv']=='v385_baseline.csv';")
    s = s.replace('warped_median_experimental', route+'_experimental').replace('recursive_metric_experimental', route+'_experimental')
    s = s.replace("'v384没有合格更新，v385仍直接比较完整v385。'", "'本轮独立比较完整v385，不与其他未评分候选叠加。'")
    s = s.replace("'v383为独立的旧脂溶性恢复文件，不混入本轮主实验。'", "'本轮独立比较完整v385，不与其他未评分候选叠加。'")
    s = s.replace("ablations=[]", "ablations=[]")
    s = s.replace("'实验提交：`'", "'实验提交：`'")
    s = s.replace("audit.append(dict(task=t,best=best['config'],weight=best['weight'],parts=parts,accepted=positive,strict_accepted=strict))",
        "audit.append(dict(task=t,best=best['config'],weight=best['weight'],parts=parts,accepted=positive,strict_accepted=strict,\n   rejection_reasons=[] if positive else ([x['part']+':MAE不改善'for x in parts if x['gain']<=0]+[x['part']+':Spearman不改善'for x in parts if x.get('rank_gain',1)<=0]+(['开发门槛未过，未运行确认']if len(parts)<4 else[]))))")
    s = s.replace("  lines.append('|'+ '|'.join([NAMES", "  lines.append('|'+ '|'.join([NAMES")
    s = s.replace("  lines.append('|'+ '|'.join([NAMES[TASKS.index(r['task'])],r['best']['name'],str(r['weight']),*[gains.get(p,'未运行')for p in PARTS],label])+'|')",
        "  lines.append('|'+ '|'.join([NAMES[TASKS.index(r['task'])],r['best']['name'],str(r['weight']),*[gains.get(p,'未运行')for p in PARTS],label])+'|')\n lines+=['','未采用原因：']\n for r in audit:\n  if r['rejection_reasons']:lines.append('- '+NAMES[TASKS.index(r['task'])]+'：'+'；'.join(r['rejection_reasons']))")
    return s


def main():
    s = common((ROOT/'tools/v385_deliver.py').read_text(encoding='utf-8'), 'v386', 'oof_monotone')
    s = s.replace("assert verified['development_fits']==296", "assert verified['development_base_models']==216\n assert sum(r['part']in PARTS[:2] and r['calibration_certificate']is not None for r in verified['records'])==108")
    s = s.replace("full_development_solves=296,all_solves_verified=verified['models']", "full_development_base_fits=216,full_development_calibration_fits=108,prediction_packages_verified=verified['models'],all_base_fits_verified=verified['base_models'],all_calibration_fits_verified=verified['calibration_models']")
    s = s.replace('296次开发求解全部完成', '216次开发基础SVR、108次开发校准LP全部完成')
    s = s.replace('v384没有合格更新，v386仍直接比较完整v385。', '本轮独立比较完整v385，不与其他未评分候选叠加。')
    (ROOT/'tools/v386_deliver.py').write_text(s, encoding='utf-8')
    s = common((ROOT/'tools/v384_deliver.py').read_text(encoding='utf-8'), 'v387', 'correlated_metric')
    (ROOT/'tools/v387_deliver.py').write_text(s, encoding='utf-8')


if __name__ == '__main__':
    main()
