"""Standard-library-only independent graph submission prediction replay."""
import csv,io,json,hashlib,math
from pathlib import Path
from collections import Counter,defaultdict
from admet_v24_local_verify import ranks
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def main():
    dest=ROOT/'admet_submissions/v27';path=next(dest.glob('*.csv'));old=next((ROOT/'admet_submissions/v24').glob('*.csv'))
    info=json.loads((dest/'verification.json').read_text(encoding='utf-8'))
    analysis=json.loads((ROOT/'admet_model_outputs_v27/analysis.json').read_text(encoding='utf-8'))
    predictions=json.loads((ROOT/'admet_model_outputs_v27/test_predictions_compact.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'admet_model_outputs_v27/source_manifest.json').read_text(encoding='utf-8'))
    assert sha(path)==info['sha256'] and sha(old)==info['baseline_sha256']
    assert info['signature']==analysis['signature']
    for name in ['admet_v27_protocol.json','admet_v27_models.py','admet_v27_run.py']:assert sha(ROOT/name)==manifest[name]
    a,c,t=read(path),read(old),read(ROOT/'admet_official/submission_template.csv')
    assert len(a)==len(c)==len(t)==16380 and a[0]==c[0]==t[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
    data=path.read_bytes();assert data.startswith(b'\xef\xbb\xbf') and data.count(b'\r\n')==data.count(b'\n')==16380
    changed=Counter();bytask=defaultdict(list);identities=set()
    for now,before,template in zip(a[1:],c[1:],t[1:]):
        assert len(now)==6 and now[:4]+now[5:]==before[:4]+before[5:]==template[:4]+template[5:]
        identity=(now[0],now[1]);assert identity not in identities;identities.add(identity)
        val=float(now[4]);assert math.isfinite(val)
        if now[5] in ['AUROC','AUPRC']:assert 0<=val<=1
        if val!=float(before[4]):changed[now[0]]+=1
        bytask[now[0]].append((float(before[4]),val))
    assert len(bytask)==22 and dict(changed)==info['changed_tasks']
    for record in analysis['tasks']:
        task=record['task'];base=[x for x,y in bytask[task]];actual=[y for x,y in bytask[task]];sel=record['selection']
        if sel is None:
            assert base==actual
            continue
        cand=predictions[task][sel['variant']]
        if record['metric']!='MAE':base,cand=ranks(base),ranks(cand)
        expected=[(1-sel['weight'])*x+sel['weight']*y for x,y in zip(base,cand)]
        assert len(expected)==len(actual) and all(abs(x-y)<1e-12 for x,y in zip(expected,actual))
    verification=dict(status='INDEPENDENT_LOCAL_REPLAY_PASSED',sha256=sha(path),rows=16379,tasks=22,columns=6,
        bytes=len(data),changed_tasks=dict(changed),bom=True,crlf=True,template_identity_exact=True,
        prediction_fusion_replayed=True,source_hashes_match=True,local_gpu_training=False)
    (dest/'local_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    cp=ROOT/'admet_current_candidate.json';state=json.loads(cp.read_text(encoding='utf-8'))
    state.update(experimental_release='v27',experimental_status='EXPERIMENTAL_AWAITING_PLATFORM',
       experimental_submission=str(path),experimental_sha256=sha(path),awaiting='User platform evaluation of v27 graph model',
       verification='admet_submissions/v27/local_verification.json',report='admet_diagnostics/v27_report.md')
    cp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
