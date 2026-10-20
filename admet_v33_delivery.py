"""Promote only verified artifacts as experimental; keep confirmed score separate."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    dest=ROOT/'admet_submissions/v33';file=next(dest.glob('*.csv'))
    verify=json.loads((dest/'local_verification.json').read_text(encoding='utf-8'))
    analysis=json.loads((ROOT/'admet_model_outputs_v33/analysis.json').read_text(encoding='utf-8'))
    assert verify['status']=='INDEPENDENT_MODEL_AND_CSV_REPLAY_PASSED'
    p=ROOT/'admet_current_candidate.json';state=json.loads(p.read_text(encoding='utf-8'))
    assert state['platform_anchor_release']=='v32' and state['platform_anchor_score']==50.399027
    state.update(experimental_release='v33',experimental_submission=str(file),experimental_sha256=verify['sha256'],
        experimental_status='EXPERIMENTAL_AWAITING_PLATFORM',candidate_platform_score=None,
        awaiting='User competition evaluation of v33 learned chemical-neighbor assay transfer',
        ready_for_platform_evaluation=True,changed_task_count=len(analysis['selected_tasks']),
        report='admet_diagnostics/v33_report.md',verification='admet_submissions/v33/local_verification.json',
        remote_model_verification='admet_model_outputs_v33/model_replay.json',
        automatically_uploaded=False,local_gpu_training=False,target_achieved=False)
    p.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');print(str(file))
if __name__=='__main__':main()
