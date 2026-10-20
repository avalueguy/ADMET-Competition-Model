"""Read-only summary of completed remote v8 work."""
import json
from collections import Counter
from pathlib import Path

root = Path('admet_model_outputs_v8')
p = json.loads(Path('admet_v8_protocol.json').read_text())
rows = []
for path in sorted(root.glob('*/decision.json')):
    d = json.loads(path.read_text())
    rows.append(dict(task=d['task'], accepted=d['accepted'], family=d['selection']['family'],
                     alpha=d['selection']['alpha'], reason=d['reason'], before=d.get('before'), after=d.get('after')))
state = json.loads((root/'coordinator_status.json').read_text()) if (root/'coordinator_status.json').exists() else {}
print(json.dumps(dict(coordinator=state,
    completed={f:len(list(root.glob('*/'+f+'_cv.json'))) for f in ['multiscale_kernel','gated_transfer']},
    completed_folds={f:len(list(root.glob('*/'+f+'_fold*.npz'))) for f in ['multiscale_kernel','gated_transfer']},
    decided=len(rows), accepted=[r for r in rows if r['accepted']],
    reasons=dict(Counter(r['reason'] for r in rows))), indent=2))
