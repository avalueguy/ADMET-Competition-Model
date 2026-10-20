"""Independent numerical fusion and exact CSV verification; lightweight arrays only."""
import csv
import hashlib
import io
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from admet_v24_local_verify import ranks
ROOT = Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))

def main():
    out = ROOT/'admet_model_outputs_v44'
    dest = ROOT/'admet_submissions/v44'
    file = next(dest.glob('*.csv'))
    basefile = next((ROOT/'admet_submissions/v43').glob('*.csv'))
    info = json.loads((dest/'verification.json').read_text(encoding='utf-8'))
    report = json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    replay = json.loads((out/'model_replay.json').read_text(encoding='utf-8'))
    compact = json.loads((out/'test_predictions_compact.json').read_text(encoding='utf-8'))
    manifest = json.loads((out/'source_manifest.json').read_text(encoding='utf-8'))
    assert info['signature']==report['signature']==replay['signature']
    assert replay['status']=='FINAL_PAIR_AND_ANCHOR_ISOLATION_REPLAY_PASSED'
    for name,digest in manifest.items():assert sha(ROOT/name)==digest
    verified = set()
    for row in replay['tasks']:
        for key in ['model_reload_passed','fit_only_anchors_rebuilt','query_cross_scaffold_anchors_checked',
                    'hidden_target_labels_invariant','saved_fit_preprocessors_match','antisymmetric_prediction_passed','query_batch_invariant']:
            assert row[key]
        verified.add(row['task'])
    assert sha(file)==info['sha256'] and sha(basefile)==info['baseline_sha256'] and sha(file)!=sha(basefile)
    data = file.read_bytes()
    assert data.startswith(b'\xef\xbb\xbf') and data.count(b'\r\n')==data.count(b'\n')==16380
    now,old,template = read(file),read(basefile),read(ROOT/'admet_official/submission_template.csv')
    assert len(now)==len(old)==len(template)==16380
    assert now[0]==old[0]==template[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
    changed = Counter();bytask=defaultdict(list);keys=set()
    for a,c,t in zip(now[1:],old[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==c[:4]+c[5:]==t[:4]+t[5:]
        key=(a[0],a[1]);assert key not in keys;keys.add(key)
        p=float(a[4]);assert math.isfinite(p)
        if a[5] in ['AUROC','AUPRC']:assert 0<=p<=1
        if p!=float(c[4]):changed[a[0]]+=1
        bytask[a[0]].append((float(c[4]),p))
    assert len(bytask)==22 and dict(changed)==info['changed_tasks']
    covered=set();ranking={}
    for r in report['tasks']:
        task=r['task'];s=r['selection'];covered.add(task)
        prior=[a for a,_ in bytask[task]];actual=[c for _,c in bytask[task]]
        if s is None:assert prior==actual;continue
        assert task in verified
        raw=compact[task]
        assert len(raw)==4 and all(len(a)==len(prior) and all(math.isfinite(v) for v in a) for a in raw)
        # Preserve the exact array reduction order used remotely. Differences of
        # 1e-16 in rank averages can split a tie before the final ranking step.
        values=np.array(raw) if r['metric']=='MAE' else np.stack([ranks(a) for a in raw])
        if s['variant']<4:candidate=raw[s['variant']]
        else:
            mean=np.mean(values[1:],axis=0)
            candidate=mean if s['variant']==4 else .5*values[0]+.5*mean
        if r['metric']!='MAE':
            ranking[task]=sum(a!=b for a,b in zip(ranks(prior),ranks(actual)))
            prior,candidate=ranks(prior),ranks(candidate)
        expected=[(1-s['weight'])*a+s['weight']*c for a,c in zip(prior,candidate)]
        assert all(abs(a-c)<1e-12 for a,c in zip(actual,expected))
    for task,values in bytask.items():
        if task not in covered:assert all(a==c for a,c in values)
    result=dict(status='INDEPENDENT_PAIR_FUSION_AND_CSV_PASSED',sha256=sha(file),rows=16379,tasks=22,columns=6,
        bytes=len(data),changed_tasks=dict(changed),bom=True,crlf=True,template_identity_exact=True,source_hashes_match=True,
        final_fusion_replayed=True,remote_model_reload=True,label_isolation_passed=True,fit_only_anchors_rebuilt=True,
        ranking_changed_rows=ranking,local_gpu_training=False)
    (dest/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
