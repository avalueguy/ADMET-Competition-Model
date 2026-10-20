"""Replay each source model mixture and final CSV using standard Python only."""
import csv,io,json,hashlib,math
from pathlib import Path
from collections import Counter,defaultdict
from admet_v24_local_verify import ranks
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def main():
    dest=ROOT/'admet_submissions/v29';file=next(dest.glob('*.csv'));old=next((ROOT/'admet_submissions/v24').glob('*.csv'))
    info=json.loads((dest/'verification.json').read_text(encoding='utf-8'));analysis=json.loads((ROOT/'admet_model_outputs_v29/analysis.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'admet_model_outputs_v29/source_manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest.items():assert sha(ROOT/name)==digest
    assert sha(file)==info['sha256'] and sha(old)==info['baseline_sha256'] and info['signature']==analysis['signature']
    data=file.read_bytes();assert data.startswith(b'\xef\xbb\xbf') and data.count(b'\r\n')==data.count(b'\n')==16380
    a,c,t=read(file),read(old),read(ROOT/'admet_official/submission_template.csv')
    assert len(a)==len(c)==len(t)==16380 and a[0]==c[0]==t[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
    bytask=defaultdict(list);changed=Counter();keys=set()
    for now,prior,template in zip(a[1:],c[1:],t[1:]):
        assert len(now)==6 and now[:4]+now[5:]==prior[:4]+prior[5:]==template[:4]+template[5:]
        key=(now[0],now[1]);assert key not in keys;keys.add(key);v=float(now[4]);assert math.isfinite(v)
        if now[5] in ['AUROC','AUPRC']:assert 0<=v<=1
        if v!=float(prior[4]):changed[now[0]]+=1
        bytask[now[0]].append((float(prior[4]),v))
    assert len(bytask)==22 and dict(changed)==info['changed_tasks']
    model_predictions={v:json.loads((ROOT/f'admet_model_outputs_v{v}/test_predictions_compact.json').read_text(encoding='utf-8')) for v in [27,28,31]}
    compact=json.loads((ROOT/'admet_model_outputs_v29/test_predictions_compact.json').read_text(encoding='utf-8'))
    for record in analysis['tasks']:
        t=record['task'];sel=record['selection'];base=[x for x,y in bytask[t]];actual=[y for x,y in bytask[t]]
        if sel is None:assert base==actual;continue
        candidate=[0.]*len(base)
        for source in sel['sources']:
            values=model_predictions[source['version']][t][source['variant']]
            if sel['name']=='assay_complement_equal' and record['metric']!='MAE':values=ranks(values)
            assert len(values)==len(base)
            candidate=[x+source['weight']*y for x,y in zip(candidate,values)]
        assert all(abs(x-y)<1e-12 for x,y in zip(candidate,compact[t]))
        if record['metric']!='MAE':base,candidate=ranks(base),ranks(candidate)
        expected=[(1-sel['weight'])*x+sel['weight']*y for x,y in zip(base,candidate)]
        assert all(abs(x-y)<1e-12 for x,y in zip(expected,actual))
    result=dict(status='INDEPENDENT_SOURCE_AND_CSV_REPLAY_PASSED',sha256=sha(file),rows=16379,columns=6,tasks=22,
       bytes=len(data),changed_tasks=dict(changed),bom=True,crlf=True,template_identity_exact=True,
       source_model_mixtures_replayed=True,final_fusion_replayed=True,source_hashes_match=True,local_gpu_training=False)
    (dest/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    cp=ROOT/'admet_current_candidate.json';state=json.loads(cp.read_text(encoding='utf-8'))
    state.update(experimental_release='v29',experimental_status='EXPERIMENTAL_AWAITING_PLATFORM',experimental_submission=str(file),
       experimental_sha256=sha(file),awaiting='User competition evaluation of v29 graph, MAE and conditional-assay ensemble',
       report='admet_diagnostics/v29_report.md',verification='admet_submissions/v29/local_verification.json')
    cp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
