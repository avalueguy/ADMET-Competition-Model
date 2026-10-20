"""Independent stdlib-only replay of v25 model fusion and official CSV contract."""
import csv,hashlib,io,json,math
from collections import Counter,defaultdict
from pathlib import Path
from admet_v24_local_verify import ranks

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))

def main():
    dest=ROOT/'admet_submissions/v25';new=next(dest.glob('*.csv'));old=next((ROOT/'admet_submissions/v24').glob('*.csv'))
    proof=json.loads((dest/'verification.json').read_text(encoding='utf-8'))
    analysis=json.loads((ROOT/'admet_model_outputs_v25/analysis.json').read_text(encoding='utf-8'))
    predictions=json.loads((ROOT/'admet_model_outputs_v25/test_predictions_compact.json').read_text(encoding='utf-8'))
    assert sha(new)==proof['sha256'] and sha(old)==proof['baseline_sha256']
    assert analysis['signature']==proof['signature']
    a,c,t=rows(new),rows(old),rows(ROOT/'admet_official/submission_template.csv')
    assert len(a)==len(c)==len(t)==16380
    assert a[0]==c[0]==t[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
    data=new.read_bytes();assert data.startswith(b'\xef\xbb\xbf')
    assert data.count(b'\r\n')==data.count(b'\n')==16380
    changed=Counter();bytask=defaultdict(list);seen=set()
    for now,before,template in zip(a[1:],c[1:],t[1:]):
        assert len(now)==6 and now[:4]+now[5:]==before[:4]+before[5:]==template[:4]+template[5:]
        key=(now[0],now[1]);assert key not in seen;seen.add(key)
        value=float(now[4]);assert math.isfinite(value)
        if now[5] in ['AUROC','AUPRC']:assert 0<=value<=1
        if value!=float(before[4]):changed[now[0]]+=1
        bytask[now[0]].append((float(before[4]),value))
    assert len(bytask)==22 and dict(changed)==proof['changed_tasks']
    for record in analysis['tasks']:
        task=record['task'];weight=record['weight']
        oldp=[a for a,b in bytask[task]];actual=[b for a,b in bytask[task]]
        if not weight:
            assert oldp==actual
            continue
        candidate=predictions[task][record['variant']]
        if record['metric']!='MAE':oldp,candidate=ranks(oldp),ranks(candidate)
        expected=[(1-weight)*x+weight*y for x,y in zip(oldp,candidate)]
        assert len(expected)==len(actual) and all(abs(x-y)<1e-12 for x,y in zip(expected,actual))
    # Fingerprint the exact source used to train this release.
    source_files=['admet_v25_models.py','admet_v25_run.py','admet_v25_protocol.json','admet_v26_run.py','admet_v26_protocol.json']
    remote=json.loads((ROOT/'admet_model_outputs_v25/source_manifest.json').read_text(encoding='utf-8'))
    for name in source_files:assert sha(ROOT/name)==remote[name]
    result=dict(status='INDEPENDENT_LOCAL_REPLAY_PASSED',sha256=sha(new),rows=16379,columns=6,tasks=22,
           bytes=len(data),changed_tasks=dict(changed),bom=True,crlf=True,template_identity_exact=True,
           prediction_fusion_replayed=True,source_hashes_match_remote=True,local_training=False)
    (dest/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    path=ROOT/'admet_current_candidate.json';state=json.loads(path.read_text(encoding='utf-8'))
    state.update(experimental_release='v25',experimental_status='EXPERIMENTAL_AWAITING_PLATFORM',
                 experimental_submission=str(new),experimental_sha256=sha(new),
                 awaiting='User platform evaluation of v25; v24 remains the confirmed scored anchor',
                 report='admet_diagnostics/v25_report.md',verification='admet_submissions/v25/local_verification.json')
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
