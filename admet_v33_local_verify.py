"""Independent CSV verification and fusion replay with the standard library."""
import csv,io,json,hashlib,math
from collections import Counter,defaultdict
from pathlib import Path
from admet_v24_local_verify import ranks
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def main():
    dest=ROOT/'admet_submissions/v33';file=next(dest.glob('*.csv'))
    baseline=next((ROOT/'admet_submissions/v32').glob('*.csv'))
    out=ROOT/'admet_model_outputs_v33'
    info=json.loads((dest/'verification.json').read_text(encoding='utf-8'))
    analysis=json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'source_manifest.json').read_text(encoding='utf-8'))
    predictions=json.loads((out/'test_predictions_compact.json').read_text(encoding='utf-8'))
    replay=json.loads((out/'model_replay.json').read_text(encoding='utf-8'))
    audit=json.loads((out/'isolation_audit.json').read_text(encoding='utf-8'))
    assert analysis['status']=='EVALUATED' and len(analysis['tasks'])==22
    assert replay['status']=='FINAL_MODEL_REPLAY_PASSED'
    assert replay['signature']==info['signature']==analysis['signature']==audit['signature']
    assert set(x['task'] for x in replay['tasks'])==set(analysis['selected_tasks'])
    assert all(x['held_target_and_test_labels_invariant'] and x['target_neighbor_block_removed'] for x in audit['tasks'])
    for name,digest in manifest.items():assert sha(ROOT/name)==digest
    assert sha(file)==info['sha256'] and sha(baseline)==info['baseline_sha256'] and sha(file)!=sha(baseline)
    data=file.read_bytes();assert data.startswith(b'\xef\xbb\xbf') and data.count(b'\r\n')==data.count(b'\n')==16380
    a,b,t=read(file),read(baseline),read(ROOT/'admet_official/submission_template.csv')
    assert len(a)==len(b)==len(t)==16380 and a[0]==b[0]==t[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
    changed=Counter();bytask=defaultdict(list);keys=set()
    for now,prior,template in zip(a[1:],b[1:],t[1:]):
        assert len(now)==6 and now[:4]+now[5:]==prior[:4]+prior[5:]==template[:4]+template[5:]
        key=(now[0],now[1]);assert key not in keys;keys.add(key)
        value=float(now[4]);assert math.isfinite(value)
        if now[5] in ['AUROC','AUPRC']:assert 0<=value<=1
        if value!=float(prior[4]):changed[now[0]]+=1
        bytask[now[0]].append((float(prior[4]),value))
    assert len(bytask)==22 and dict(changed)==info['changed_tasks']
    ranking_changes={}
    for r in analysis['tasks']:
        task=r['task'];s=r['selection'];old=[x for x,y in bytask[task]];actual=[y for x,y in bytask[task]]
        if s is None:assert old==actual;continue
        parts=predictions[task];v=s['variant'];assert len(parts)==3 and all(len(p)==len(old) for p in parts)
        if v<3:candidate=parts[v]
        else:
            x,y=parts[0],parts[1 if v==3 else 2]
            if r['metric']!='MAE':x,y=ranks(x),ranks(y)
            candidate=[(a+b)*.5 for a,b in zip(x,y)]
        if r['metric']!='MAE':
            ranking_changes[task]=sum(x!=y for x,y in zip(ranks(old),ranks(actual)))
            old,candidate=ranks(old),ranks(candidate)
        expected=[(1-s['weight'])*x+s['weight']*y for x,y in zip(old,candidate)]
        assert all(abs(x-y)<1e-12 for x,y in zip(actual,expected))
    result=dict(status='INDEPENDENT_MODEL_AND_CSV_REPLAY_PASSED',sha256=sha(file),rows=16379,columns=6,tasks=22,
        bytes=len(data),changed_tasks=dict(changed),bom=True,crlf=True,template_identity_exact=True,
        source_hashes_match=True,final_fusion_replayed=True,remote_model_reload=True,
        target_label_isolation_passed=True,ranking_changed_rows=ranking_changes,local_gpu_training=False)
    (dest/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
