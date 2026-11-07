"""Reconstruct v114's v68 half-life branch for a matched valid comparison.

Read only archived predictions and official development labels. This audit
does not select model configurations or blend weights.
"""
import csv, hashlib, io, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v115_diverse_specialists as sequence
import v117_conditional_structure as conditional
b,r83=sequence.b,sequence.r83

def archive(path):
    manifest=json.loads((ROOT/'v118_original_manifest.json').read_text())
    expected=next(r['sha256'] for r in manifest['files'] if r['path']==path)
    assert b.sha(ROOT/path)==expected,path
    return np.load(ROOT/path)

def baseline(d,k,part):
    q=r83.split(d,k,part)[1];neural=[]
    for seed in (672031,672032):
        z=archive(f'admet_model_outputs_v67/{part}/shared/{seed}/predictions.npz')
        pos={int(i):j for j,i in enumerate(z['query'])}
        neural.append(z['p'][[pos[int(i)] for i in q]])
    z=archive(f'admet_model_outputs_v68/half_life_obach/{"valid_report" if part=="valid" else part}/predictions.npz')
    np.testing.assert_array_equal(q,z['query'])
    return .65*r83.rankdata(np.mean(neural,axis=0))/len(q)+.35*r83.rankdata(z['p'][2])/len(q)

def run():
    d=sequence.load();k=d['tasks'].index('half_life_obach')
    # The final historical prediction must be exactly the carried v114 branch.
    z=baseline(d,k,'final')
    file=next((ROOT/'admet_submissions/v114_herg_history52').glob('*.csv'))
    rows=list(csv.DictReader(io.StringIO(file.read_bytes().decode('utf-8-sig'),newline='')))
    actual=np.array([float(r['y_pred']) for r in rows if r['task']=='half_life_obach'])
    np.testing.assert_allclose(z,actual,atol=1e-14,rtol=0)
    a=baseline(d,k,'valid');q=r83.split(d,k,'valid')[1];y=d['y'][q]
    records=[]
    for name,v in [('v115',sequence),('v117',conditional)]:
        path=v.OUT/'confirmation.json'
        if not path.exists():continue
        doc=json.loads(path.read_text())
        if doc['status']!='COMPLETE':continue
        assert doc['signature']==v.sig()
        for r in doc['records']:
            if r['task']!='half_life_obach':continue
            expert=sequence.predict(d,k,'valid',r['config']) if v is sequence else conditional.fit(d,k,'valid',r['depth'],'full')
            p=(1-v.WEIGHT)*r83.rankdata(a)/len(a)+v.WEIGHT*r83.rankdata(expert)/len(expert)
            records.append(dict(version=name,baseline=r83.score('Spearman',y,a),
                expert=r83.score('Spearman',y,expert),candidate=r83.score('Spearman',y,p),
                gain=r83.gain('Spearman',y,a,p),original_confirmation_accepted=r['accepted']))
    report=dict(status='PASS',task='half_life_obach',final_baseline_replay_exact=True,
        matched_official_valid_records=records,rows=len(q),source_manifest_sha256=b.sha(ROOT/'v118_original_manifest.json'),
        limitations='Official valid labels historically reused. CV confirmation remains relative to approximate v101 proxy; this is an additional exact-branch check.')
    b.save(ROOT/'admet_model_outputs_v118/exact_half_life_audit.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':run()
