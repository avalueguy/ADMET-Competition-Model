"""Read complete development predictions only; never fit or change selection."""
from pathlib import Path
import json
import numpy as np
import v375_stereo_context as m


def main():
    d = m.load()
    records = []
    for task in m.TASKS:
        paths = [m.OUT / 'tasks' / task / part / c['name'] / 'predictions.npz'
                 for part in ('fold0', 'fold1') for c in m.CONFIGS]
        if not all(p.exists() for p in paths):
            continue
        options = []
        for c in m.CONFIGS:
            for weight in m.WEIGHTS:
                gains = []
                for part in ('fold0', 'fold1'):
                    a = m.ctrl(task, part)
                    z = np.load(m.OUT / 'tasks' / task / part / c['name'] / 'predictions.npz')
                    assert str(z['signature']) == m.signature()
                    y = d['y'][a['query']]
                    old = np.abs(y - a['p']).mean()
                    candidate = (1-weight) * a['p'] + weight * z['p']
                    gains.append(float((old - np.abs(y-candidate).mean()) / old))
                options.append(dict(task=task, config=c['name'], weight=weight,
                                    gains=gains, objective=min(gains)))
        best = max(options, key=lambda r: (r['objective'], -r['weight'],
                    -[x['name'] for x in m.CONFIGS].index(r['config'])))
        records.append(best)
    value = dict(stage='DEVELOPMENT_SNAPSHOT_ONLY', records=records, no_fitting=True,
                 no_selection_mutation=True, confirmation_unread=True)
    (Path(__file__).resolve().parent / 'development_snapshot.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
