"""Generate non-model launch/receipt helpers before immutable freeze."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    for name in ('bootstrap','completion','monitor'):
        source=(ROOT/f'tools/v380_{name}.py').read_text(encoding='utf-8')
        source=source.replace('regression9_v380','regression9_v381').replace('v380_smoke','v381_smoke').replace('v380_run_experts','v381_run_experts').replace('v380_verify_experts','v381_verify_experts').replace('v380_histogram_median','v381_oof_pareto').replace('v380 runner','v381 runner')
        if name=='bootstrap':
            source=source.replace("regression9_v379/completion_receipt.json","regression9_v380/completion_receipt.json").replace('workers=3','workers=2')
        if name=='monitor':
            source=source.replace("result['total_development_fits']=104","result['total_development_prediction_packages']=52\n    result['total_development_base_models']=216\n    result['completed_development_base_models']=3*sum(p.parts[-3] in ('fold0','fold1') for p in (ROOT/'outputs/base').glob('*/*/*/state.joblib'))")
        (ROOT/f'tools/v381_{name}.py').write_text(source,encoding='utf-8')


if __name__=='__main__':main()
