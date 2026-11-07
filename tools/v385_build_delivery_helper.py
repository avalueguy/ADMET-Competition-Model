"""Adapt the no-fit receiver, preserving v384 sources and evidence."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 s=(ROOT/'tools/v384_deliver.py').read_text(encoding='utf-8').replace('v384','v385').replace('V384','V385').replace('recursive_metric','warped_median').replace('324','296')
 s=s.replace("assert sum(9 for r in verified['curve_records']if r['part']in PARTS[:2])==296","assert verified['development_fits']==296")
 start=s.index('  byround=[]');end=s.index(' means=',start);s=s[:start]+s[end:]
 s=s.replace("save(ROOT/'docs/v385_delivery.json',result);save(ROOT/'docs/v385_learning_curves.json',dict(signature=sig,ablations=ablations,all_fixed_horizons_completed=True,no_confirmation_reselection=True))","save(ROOT/'docs/v385_delivery.json',result)")
 s=s.replace('v383为独立的旧脂溶性恢复文件，不混入本轮主实验。','v384没有合格更新，v385仍直接比较完整v382。')
 (ROOT/'tools/v385_deliver.py').write_text(s,encoding='utf-8')
if __name__=='__main__':main()
