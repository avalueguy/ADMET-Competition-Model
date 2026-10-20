"""Independently replay v26 serialization and official submission identity."""
import csv,hashlib,io,json,math
from collections import Counter,defaultdict
from pathlib import Path
from admet_v24_local_verify import ranks

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))

def main():
    dest=ROOT/'admet_submissions/v26';new=next(dest.glob('*.csv'))
    old=next((ROOT/'admet_submissions/v24').glob('*.csv'))
    proof=json.loads((dest/'verification.json').read_text(encoding='utf-8'))
    analysis=json.loads((ROOT/'admet_model_outputs_v26/analysis.json').read_text(encoding='utf-8'))
    pred=json.loads((ROOT/'admet_model_outputs_v26/test_predictions_compact.json').read_text(encoding='utf-8'))
    assert sha(new)==proof['sha256'] and sha(old)==proof['baseline_sha256']
    assert analysis['signature']==proof['signature'] and proof['baseline']=='v24'
    a,b,t=read(new),read(old),read(ROOT/'admet_official/submission_template.csv')
    assert len(a)==len(b)==len(t)==16380
    assert a[0]==b[0]==t[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
    data=new.read_bytes();assert data.startswith(b'\xef\xbb\xbf')
    assert data.count(b'\r\n')==data.count(b'\n')==16380
    changed=Counter();baseline={};actual={}
    for now,prior,template in zip(a[1:],b[1:],t[1:]):
        assert len(now)==6 and now[:4]+now[5:]==prior[:4]+prior[5:]==template[:4]+template[5:]
        value=float(now[4]);assert math.isfinite(value)
        if now[5] in ['AUROC','AUPRC']:assert 0<=value<=1
        if value!=float(prior[4]):changed[now[0]]+=1
        baseline.setdefault(now[0],[]).append(float(prior[4]));actual.setdefault(now[0],[]).append(value)
    assert dict(changed)==proof['changed_tasks']
    for record in analysis['records']:
        task=record['task'];source=baseline[task].copy();candidate=pred[task][record['variant']]
        if record['metric']!='MAE':source,candidate=ranks(source),ranks(candidate)
        expected=[(1-record['weight'])*x+record['weight']*y for x,y in zip(source,candidate)]
        assert len(expected)==len(actual[task]) and all(abs(x-y)<1e-12 for x,y in zip(expected,actual[task]))
    manifest=json.loads((ROOT/'admet_model_outputs_v25/source_manifest.json').read_text(encoding='utf-8'))
    for filename in ['admet_v25_models.py','admet_v25_run.py','admet_v25_protocol.json','admet_v26_run.py','admet_v26_protocol.json']:
        assert sha(ROOT/filename)==manifest[filename]
    result=dict(status='INDEPENDENT_LOCAL_REPLAY_PASSED',sha256=sha(new),rows=16379,columns=6,tasks=22,
        bytes=len(data),changed_tasks=dict(changed),bom=True,crlf=True,template_identity_exact=True,
        prediction_fusion_replayed=True,remote_source_hashes_match=True,local_training=False)
    (dest/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    state_path=ROOT/'admet_current_candidate.json';state=json.loads(state_path.read_text(encoding='utf-8'))
    state.update(recommended_release='v24',submission=str(old),sha256=sha(old),platform_anchor_release='v24',
       platform_anchor_score=50.248759,candidate_platform_score=50.248759,experimental_release='v26',
       experimental_status='EXPERIMENTAL_AWAITING_PLATFORM',experimental_submission=str(new),experimental_sha256=sha(new),
       awaiting='User platform evaluation of v26; v24 remains the confirmed scored anchor',
       report='admet_diagnostics/v26_report.md',verification='admet_submissions/v26/local_verification.json')
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
