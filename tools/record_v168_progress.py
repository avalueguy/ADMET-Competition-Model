"""Persist actual training state and explicit Edge submission authorization."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    state=json.loads((ROOT/'CURRENT_STATUS.json').read_text(encoding='utf-8-sig'))
    state.update(active_experiment='v168_sparse_linear',active_experiment_status='TRAINING_AND_FROZEN_CONFIRMATION',new_submission_created=False,running_new_training_processes=1,
        priority_next_step='Finish v168 unchanged strict protocol, v169 replay/export; archive locally and submit changed verified CSV through authorized Edge. No candidate means no upload.')
    state['v161_result']=dict(status='NO_CONFIRMED_NEW_MODEL',routes=[149,150,151,153,157,159],accepted_tasks=[],submission_created=False,official_score=None,baseline_retained='v145',archive='releases/v161_new_structure_models_20261027.tar.gz',archive_verification='IN_PROGRESS')
    state['v168_training']=dict(status='RUNNING',source='tools/v168_sparse_linear.py',source_sha256=hashlib.sha256((ROOT/'tools/v168_sparse_linear.py').read_bytes()).hexdigest(),remote_pid=16560,verification_pid=16652,
        baseline='v145',method='Sparse five-view TF-IDF/NB linear models with fit-only preprocessing; target-excluded augmented chemistry; independent task fitting',protocol='docs/v168_training_plan.json',acceptance_rules_relaxed=False,official_score=None)
    state['platform_followup_automation']=dict(id='admet',status='ACTIVE',kind='heartbeat',interval_minutes=15,notification='New result, completion, failure or user action only')
    save(ROOT/'CURRENT_STATUS.json',state)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8-sig');text=text.replace('六条新模型路线正在运行，候选编号 v161，尚未生成新提交。','六条新模型路线 v161 已完成，未通过确认门槛，没有生成或上传新提交。新增 v168 稀疏线性路线正在训练，v169 负责独立重载验证与导出；自动跟进已启用。',1);readme.write_text(text,encoding='utf-8')
    print('RECORDED_V168_RUNNING')
if __name__=='__main__':main()
