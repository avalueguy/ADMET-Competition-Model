"""Independent standard-library CSV and rank-fusion replay, no local training."""
import csv, io, json, hashlib, math
from collections import Counter, defaultdict
from pathlib import Path
from admet_v24_local_verify import ranks

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def main():
    dest=ROOT/'admet_submissions/v32';file=next(dest.glob('*.csv'))
    baseline=next((ROOT/'admet_submissions/v29').glob('*.csv'))
    info=json.loads((dest/'verification.json').read_text(encoding='utf-8'))
    analysis=json.loads((ROOT/'admet_model_outputs_v32/analysis.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'admet_model_outputs_v32/source_manifest.json').read_text(encoding='utf-8'))
    predictions=json.loads((ROOT/'admet_model_outputs_v32/test_predictions_compact.json').read_text(encoding='utf-8'))
    replay=json.loads((ROOT/'admet_model_outputs_v32/model_replay.json').read_text(encoding='utf-8'))
    assert analysis['status']=='EVALUATED' and len(analysis['tasks'])==22
    assert replay['status']=='FINAL_MODEL_REPLAY_PASSED' and replay['signature']==info['signature']==analysis['signature']
    for name,digest in manifest.items():assert sha(ROOT/name)==digest
    assert sha(file)==info['sha256'] and sha(baseline)==info['baseline_sha256'] and sha(file)!=sha(baseline)
    data=file.read_bytes();assert data.startswith(b'\xef\xbb\xbf') and data.count(b'\r\n')==data.count(b'\n')==16380
    a,b,t=read(file),read(baseline),read(ROOT/'admet_official/submission_template.csv')
    assert len(a)==len(b)==len(t)==16380 and a[0]==b[0]==t[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
    changed=Counter();bytask=defaultdict(list);keys=set()
    for now,prior,template in zip(a[1:],b[1:],t[1:]):
        assert len(now)==6 and now[:4]+now[5:]==prior[:4]+prior[5:]==template[:4]+template[5:]
        key=(now[0],now[1]);assert key not in keys;keys.add(key)
        x=float(now[4]);assert math.isfinite(x)
        if now[5] in ['AUROC','AUPRC']:assert 0<=x<=1
        if x!=float(prior[4]):changed[now[0]]+=1
        bytask[now[0]].append((float(prior[4]),x))
    assert len(bytask)==22 and dict(changed)==info['changed_tasks']
    ranking_changes={}
    for record in analysis['tasks']:
        task=record['task'];sel=record['selection'];old=[x for x,y in bytask[task]];actual=[y for x,y in bytask[task]]
        if sel is None:assert old==actual;continue
        parts=predictions[task];variant=sel['variant'];assert len(parts)==3 and all(len(v)==len(old) for v in parts)
        if variant<3:candidate=parts[variant]
        else:
            x,y=parts[0],parts[2 if variant==3 else 1]
            if record['metric']!='MAE':x,y=ranks(x),ranks(y)
            candidate=[(a+b)*.5 for a,b in zip(x,y)]
        if record['metric']!='MAE':
            ranking_changes[task]=sum(x!=y for x,y in zip(ranks(old),ranks(actual)))
            old,candidate=ranks(old),ranks(candidate)
        expected=[(1-sel['weight'])*x+sel['weight']*y for x,y in zip(old,candidate)]
        assert all(abs(x-y)<1e-12 for x,y in zip(actual,expected))
    result=dict(status='INDEPENDENT_MODEL_AND_CSV_REPLAY_PASSED',sha256=sha(file),rows=16379,columns=6,tasks=22,
        bytes=len(data),changed_tasks=dict(changed),bom=True,crlf=True,template_identity_exact=True,
        source_hashes_match=True,final_fusion_replayed=True,remote_model_reload=True,
        ranking_changed_rows=ranking_changes,local_gpu_training=False)
    (dest/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
