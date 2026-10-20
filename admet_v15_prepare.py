"""Record the user's scored anchor and explicit removal of per-metric vetoes."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,x):Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    anchor=next(Path('admet_submissions/v14').glob('*.csv'))
    digest=hashlib.sha256(anchor.read_bytes()).hexdigest()
    assert digest=='97c3eb59a65585cd897e843d990940bd2a7ac67ef8ea825f7a70598101a1fd94'
    score=dict(version='v14',reported_score=49.419624,previous_score=49.381036,latest_improvement=.038588,
        maximum_published=65.719642,target=51.584929,gap_to_target=2.165305,
        next_version_minimum_target=50.019624,minimum_requested_gain=.6,
        submitted_sha256=digest,source='User report after recommended v14; linkage inferred from conversation, platform hash not available',
        aggregation_formula='unknown',per_task_platform_scores='unavailable')
    history=read('admet_diagnostics/platform_score_history.json')
    existing=[x for x in history if x.get('version')=='v14']
    if existing:assert existing[0]==score
    else:history.append(score)
    save('admet_diagnostics/platform_score_history.json',history);save('admet_diagnostics/platform_score.json',score)
    current=read('admet_current_candidate.json')
    current.update(platform_anchor_release='v14',platform_anchor_score=49.419624,candidate_platform_score=49.419624,
        next_version_minimum_target=50.019624,target_achieved=False,awaiting='v15 remote total-primary-metric optimization',
        constraint_change='User explicitly removed all per-task, secondary, fold and subgroup non-regression vetoes on 2026-09-26')
    save('admet_current_candidate.json',current)
    p=dict(version=15,created_at_utc=datetime.now(timezone.utc).isoformat(),baseline_release='v14',baseline_score=49.419624,
        baseline_sha256=digest,requested_gain=.6,next_target=50.019624,ultimate_target=51.584929,
        user_constraint_change='Per-metric no-decrease constraint explicitly removed. Optimize aggregate expected primary utility; report regressions without veto.',
        training_location='remote_only',automatic_submission=False,official_aggregation_formula=None,
        candidate_versions=list(range(6,14)),max_candidates_per_task=2,
        shortlist='All available primitive v6-v13 OOF models; rank by mean primary utility of 50% raw blend to v14; choose first, then highest remaining with absolute OOF Spearman < .995 to first, otherwise next. No secondary or worst-fold veto.',
        grid=[0,.25,.5,.75,1],modes=['raw','rank'],
        objective=dict(oof_weight=.65,official_valid_weight=.35,fold_standard_deviation_penalty=.05,weight_penalty=.0005),
        primary_scale='MAE improvement divided by v14 MAE within each split; AUROC/AUPRC/Spearman absolute improvement. Equal task weights, a surrogate only.',
        rank_mode='For ranking metrics only: within each prediction partition map each model to average percentile rank. Classification outputs percentile scores; Spearman maps combined percentile through v14 query prediction quantiles. No target labels in rank mapping.',
        selection='Two OOF-shortlisted families refit on target train for valid. Choose convex weights and raw/rank mode by aggregate primary surrogate. No individual-fold, valid, secondary, distant, or small-coverage veto. Baseline remains in grid.',
        extra_split=dict(seed=692509,fraction=.25,role='Report-only repeated scaffold diagnostic, no selection or veto'),
        model_recipes='Reuse frozen historical base model recipes; new ensemble weights learned, then selected bases retrained on train+valid and saved for independent replay.',
        limits=['Historical OOF/valid/repeat reused; all reported scores are selection-biased.',
            'Meta cross-fold check is diagnostic, not nested independent CV: cached base models can include other folds in training.',
            'No external assay labels, blind test labels, or answer lookup. No automatic platform upload.',
            'No reliable conversion from surrogate to platform points. No guarantee of +0.6 or hidden-score increase.'])
    path=Path('admet_v15_protocol.json')
    if path.exists():raise RuntimeError('Refuse to overwrite an existing v15 protocol')
    save(path,p)
    print(json.dumps(dict(anchor_score=49.419624,delta=.038588,next_target=50.019624)))


if __name__=='__main__':main()
