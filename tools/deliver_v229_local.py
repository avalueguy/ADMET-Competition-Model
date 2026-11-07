"""Preserve the frozen finalizer, correcting only its CURRENT_STATUS release key.

The v225 template left a literal '225' in the reporting dictionary. Training,
acceptance, model replay, CSV bytes and every assertion remain frozen.
"""
import hashlib
from pathlib import Path
import finalize_v229_local as frozen


def main():
    root = Path(__file__).resolve().parents[1]
    expected = frozen.load(root / 'docs/v228_frozen_sources.json')
    original = next(r for r in expected['files'] if r['path'] == 'tools/finalize_v229_local.py')
    assert frozen.sha(root / original['path']) == original['sha256']
    previous = frozen.load(root / 'CURRENT_STATUS.json').get('releases', {}).get('225')
    save = frozen.save

    def corrected_save(path, value):
        if Path(path).resolve() == (root / 'CURRENT_STATUS.json').resolve():
            delivery = value['releases']['225']
            assert delivery['version'] == 'v229'
            value['releases']['229'] = delivery
            if previous is None:
                value['releases'].pop('225')
            else:
                value['releases']['225'] = previous
        save(path, value)

    frozen.save = corrected_save
    frozen.main()


if __name__ == '__main__':
    main()
