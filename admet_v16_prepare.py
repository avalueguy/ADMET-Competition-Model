"""Record scored v15 and freeze the next experiment before model evaluation."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

v=read('admet_current_candidate.json');baseline=Path(v['submission'])
assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='39d93d4a0ee56b7d09d9280d9e0958c7dcaf604b63a8a7501da0ebdef2b6d8dd'
score=dict(version='v15',reported_score=49.750495,previous_score=49.419624,latest_improvement=.330871,
    maximum_published=65.719642,target=51.584929,gap_to_target=1.834434,
    submitted_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
    source='User reports score after recommended v15; file linkage inferred from conversation; platform hash unavailable',
    aggregation_formula='unknown',per_task_platform_scores='unavailable')
history=read('admet_diagnostics/platform_score_history.json')
old=[r for r in history if r.get('version')=='v15']
if old:assert old[0]==score
else:history.append(score)
save('admet_diagnostics/platform_score_history.json',history);save('admet_diagnostics/platform_score.json',score)
v.update(platform_anchor_release='v15',platform_anchor_score=49.750495,candidate_platform_score=49.750495,
    awaiting='v16 remote experiments',latest_official_gain=.330871)
save('admet_current_candidate.json',v)
p=dict(version=16,created_at_utc=datetime.now(timezone.utc).isoformat(),baseline_release='v15',baseline_score=49.750495,
    baseline_sha256=score['submitted_sha256'],next_target=50.019624,ultimate_target=51.584929,
    training_location='remote_only',automatic_submission=False,per_metric_nonregression_required=False,
    tasks=read('admet_v13_protocol.json')['tasks'],new_families=['view_attention','mol_boost'],
    seeds=[16201,16202],neural_epochs=70,neural_patience=4,neural_check_every=5,hidden=96,batch_size=128,
    learning_rate=.0007,weight_decay=.025,view_dropout=.15,
    boost_depths=[5,6],boost_max_iterations=1000,boost_patience=90,
    shortlist='Two new trained families for every task plus one historical primitive not in v15 shortlist, best mean half-blend OOF main-metric gain across raw/rank. Historical selection fixed before new fits. No per-task or secondary vetoes.',
    grid=[0,.25,.5,.75,1],modes=['raw','rank'],
    objective=dict(oof_weight=.65,official_valid_weight=.35,fold_standard_deviation_penalty=.05,weight_penalty=.0005),
    primary_scale='Relative MAE improvement or absolute AUROC/AUPRC/Spearman improvement, tasks equally weighted surrogate only',
    architecture='Four-view input-dependent gated residual neural network with random view dropout and two seeds; pretrained embedding CatBoost heads with two depths/seeds. Fit-only inner scaffold early stopping, refit on full fitting rows.',
    extra_split=dict(seed=692509,fraction=.25,role='Report-only reused scaffold diagnostic'),
    limits=['OOF, official validation and repeat split reused across versions; not unbiased generalization estimates.',
        'Official validation participates in ensemble selection. Meta leave-one-fold diagnostic is not independent nested CV.',
        'No external assay labels, blind labels, or answer lookup. Only official train/valid labels for final refit.',
        'No calibrated mapping to official score; no guaranteed improvement. v15 preserved.'])
path=Path('admet_v16_protocol.json')
assert not path.exists(),'Refuse protocol overwrite'
save(path,p);print(json.dumps(score))
