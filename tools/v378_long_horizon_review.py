"""Additional primary references and fixed-budget diagnostics; no fitting."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
from v376_material_review import fetch

ROOT=Path(__file__).resolve().parents[1]
SOURCES=(
    ('Therapeutics Data Commons evaluation framework','https://arxiv.org/abs/2102.09548'),
    ('Deep ensembles uncertainty','https://arxiv.org/abs/1612.01474'),
    ('Supervised contrastive learning','https://arxiv.org/abs/2004.11362'),
    ('Tree models versus deep learning on typical tabular data','https://arxiv.org/abs/2207.08815'),
    ('SMILES Transformer representation','https://arxiv.org/abs/1911.04738'),
)


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources=list(pool.map(fetch,SOURCES))
    value=dict(scope='Five additional selected primary pages; not exhaustive literature search.',research=sources,
               no_external_weights_downloaded=True,no_fitting=True,
               methodological_implications=[
                   'Use task metrics and structural splits; random splits alone can hide distribution shift.',
                   'Independent seeds reduce estimator variance but do not fix systematic bias or missing assay context.',
                   'Contrastive and SMILES augmentation provide views of existing molecules, not independent new labels.',
                   'Typical small tabular datasets require strong tree baselines and matched tuning budgets.',
                   'Sequence/graph pretraining benefits cannot be assumed without matching data and allowed source conditions.'
               ])
    (ROOT/'docs/v378_additional_research.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k in ('topic','title','url','status','error')} for r in sources],ensure_ascii=False))


if __name__=='__main__':main()
