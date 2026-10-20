"""Independent standard-library final-source fusion replay and CSV checks."""
import csv,io,json,hashlib,math
from collections import Counter,defaultdict
from pathlib import Path
from admet_v24_local_verify import ranks
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def main():
    dest=ROOT/'admet_submissions/v36';file=next(dest.glob('*.csv'));baseline=next((ROOT/'admet_submissions/v32').glob('*.csv'));out=ROOT/'admet_model_outputs_v36'
    info=json.loads((dest/'verification.json').read_text(encoding='utf-8'));report=json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'source_manifest.json').read_text(encoding='utf-8'));compact=json.loads((out/'test_predictions_compact.json').read_text(encoding='utf-8'))
    replay=json.loads((out/'model_replay.json').read_text(encoding='utf-8'));audit=json.loads((ROOT/'admet_model_outputs_v33/isolation_audit.json').read_text(encoding='utf-8'))
    assert info['signature']==report['signature']==replay['signature'] and replay['status']=='FINAL_MODEL_REPLAY_PASSED'
    verified={(r['task'],r['version']) for r in replay['sources'] if r['model_reload_passed']}
    for name,digest in manifest.items():assert sha(ROOT/name)==digest
    assert all(r['held_target_and_test_labels_invariant'] and r['target_neighbor_block_removed'] for r in audit['tasks'])
    assert sha(file)==info['sha256'] and sha(baseline)==info['baseline_sha256'] and sha(file)!=sha(baseline)
    data=file.read_bytes();assert data.startswith(b'\xef\xbb\xbf') and data.count(b'\r\n')==data.count(b'\n')==16380
    now,prior,template=read(file),read(baseline),read(ROOT/'admet_official/submission_template.csv')
    assert len(now)==len(prior)==len(template)==16380 and now[0]==prior[0]==template[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
    changed=Counter();bytask=defaultdict(list);keys=set()
    for a,b,t in zip(now[1:],prior[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==b[:4]+b[5:]==t[:4]+t[5:]
        key=(a[0],a[1]);assert key not in keys;keys.add(key)
        value=float(a[4]);assert math.isfinite(value)
        if a[5] in ['AUROC','AUPRC']:assert 0<=value<=1
        if value!=float(b[4]):changed[a[0]]+=1
        bytask[a[0]].append((float(b[4]),value))
    assert len(bytask)==22 and dict(changed)==info['changed_tasks'];rankchanges={}
    for r in report['tasks']:
        task=r['task'];s=r['selection'];old=[a for a,b in bytask[task]];actual=[b for a,b in bytask[task]]
        if s is None:assert old==actual;continue
        parts=compact[task];assert len(parts)==len(s['sources']);candidate=[0.]*len(old)
        for part,source in zip(parts,s['sources']):
            assert {k:part[k] for k in ['version','variant','weight']}==source and (task,source['version']) in verified
            values=part['predictions'];assert len(values)==len(old)
            if s['name']=='complement_equal' and r['metric']!='MAE':values=ranks(values)
            candidate=[a+source['weight']*b for a,b in zip(candidate,values)]
        if r['metric']!='MAE':
            rankchanges[task]=sum(a!=b for a,b in zip(ranks(old),ranks(actual)));old,candidate=ranks(old),ranks(candidate)
        expected=[(1-s['weight'])*a+s['weight']*b for a,b in zip(old,candidate)]
        assert all(abs(a-b)<1e-12 for a,b in zip(actual,expected))
    result=dict(status='INDEPENDENT_MODEL_AND_CSV_REPLAY_PASSED',sha256=sha(file),rows=16379,tasks=22,columns=6,bytes=len(data),
        changed_tasks=dict(changed),bom=True,crlf=True,template_identity_exact=True,source_hashes_match=True,final_fusion_replayed=True,
        remote_model_reload=True,target_label_isolation_passed=True,ranking_changed_rows=rankchanges,local_gpu_training=False)
    (dest/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
