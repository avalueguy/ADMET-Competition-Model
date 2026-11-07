"""Predeclare whole-task composition before v387 development scores exist."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT/'docs/v388_union_protocol.json'
    assert not (ROOT/'experiments/regression9_v387/outputs/development.json').exists()
    value = dict(
        registered_at_utc=datetime.now(timezone.utc).isoformat(),
        baseline='docs/current_best_after_v385_20261106.json',
        baseline_csv_sha256='355d8cf8f67bfd1e81b67f99e476a9780f1737d1e598a795d651bc589eeff1d0',
        source_priority=['v386','v387'],
        eligibility='Use only each frozen experiment experimental_eligible tasks; all four MAE gains positive, plus all four Spearman gains positive for ranking tasks.',
        overlap_rule='v386 wins any overlap regardless of relative validation gains; calibration route has fixed priority. No averaging, weight search or molecule routing.',
        union_rule='Whole-task replacement from the first eligible source. Source predictions and weights remain unchanged.',
        emit_rule='Create v388 only if v387 adds at least one nonoverlapping task to v386; otherwise use the original eligible experiment CSV.',
        new_official_score=None,
        repeated_public_partitions_are_not_blind=True,
    )
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        assert {k:v for k,v in old.items() if k!='registered_at_utc'} == {k:v for k,v in value.items() if k!='registered_at_utc'}
    else:
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))


if __name__=='__main__':
    main()
