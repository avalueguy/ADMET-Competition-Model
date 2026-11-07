"""Focused kernel-method references, no data/weight downloads or fitting."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from v376_material_review import fetch

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ('Pairwise kernels and feature maps', 'https://scikit-learn.org/stable/modules/metrics.html'),
    ('SVR epsilon and convergence contract', 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html'),
)


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(fetch, SOURCES))
    value = dict(scope='Two focused primary software references, not exhaustive research.',
                 research=records, no_fitting=True, external_data_downloaded=False,
                 implications=[
                     'A dot product of square-root L1-normalized nonnegative histograms is a valid explicit-map kernel.',
                     'A Gaussian kernel on the same square-root map is PSD. This differs from historical count Tanimoto.',
                     'epsilon=0 targets uninsensitive absolute residuals with RKHS regularization; finite tolerance is still approximate.',
                     'A new kernel has no guaranteed MAE gain. Convergence status and held-group performance must be checked.'
                 ])
    (ROOT / 'docs/v380_primary_references.json').write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k in ('topic','status','title','error')} for r in records]))


if __name__ == '__main__':
    main()
