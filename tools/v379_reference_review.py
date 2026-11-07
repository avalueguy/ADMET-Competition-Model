"""Selected primary sources for averaging and quantile objectives; no fitting."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
from v376_material_review import fetch

ROOT=Path(__file__).resolve().parents[1]
SOURCES=(('Mean Teachers parameter averaging','https://arxiv.org/abs/1703.01780'),
         ('Stochastic Weight Averaging','https://arxiv.org/abs/1803.05407'),
         ('Quantile Regression Forests','https://jmlr.org/papers/v7/meinshausen06a.html'))


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(fetch,SOURCES))
    report=dict(scope='Three selected original sources, no exhaustive-literature claim.',research=results,
                interpretations=[
                    'EMA is parameter smoothing. v379 is not a Mean Teacher semi-supervised reproduction and uses no teacher pseudo labels.',
                    'SWA and EMA have different averaging schedules; the SWA paper motivates averaging research, not a claim that v379 implements SWA.',
                    'Pinball loss estimates conditional quantiles. Conditional median targets MAE; auxiliary tails do not guarantee a better median or calibrated intervals.'
                ],external_weights_downloaded=False,all_fitting_remote=True)
    (ROOT/'docs/v379_primary_references.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k in ('topic','title','url','status','error')} for r in results],ensure_ascii=False))


if __name__=='__main__':main()
