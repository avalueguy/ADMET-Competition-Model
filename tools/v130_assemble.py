"""Assemble confirmed low-task experts with byte-exact protected task outputs."""
import csv,io,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v125_low_task_baseline as v
import v126_dual_target_clearance as dual
import v127_low_classification as tree
import v128_conditional_margin as kernel
from repack_upload_template import field_spans
from verify_submission import verify
b=v.b
OUT=ROOT/'admet_model_outputs_v130'
def rows(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def main():
    audit=json.loads((ROOT/'admet_model_outputs_v129/independent_verification.json').read_text())
    assert audit['status']=='PASS' and audit['baseline_sha256']==v.ANCHOR_SHA
    selected=audit['selected'];assert selected,'No supported new model to submit'
    assert set(selected)<=set(v.TASKS)
    d=v.load();original=rows(v.ANCHOR);result=[r.copy() for r in original];checks=[]
    for task,ch in selected.items():
        module={'v126':dual,'v127':tree,'v128':kernel}[ch['version']];conf=json.loads((module.OUT/'confirmation.json').read_text())
        assert conf['status']=='COMPLETE' and conf['signature']==module.sig()
        assert next(r for r in conf['records'] if r['task']==task)==ch['record'] and ch['record']['accepted']
        k=d['tasks'].index(task);pos=[i for i,r in enumerate(result) if i and r[0]==task]
        for part in ('fold0','fold1','fold2','valid','final'):
            expert=module.predict(d,k,part,ch['record']['config'] if module is kernel else ch['record']['view'])
            a=v.baseline(d,k,part);p=v.mix(d['metrics'][k],a,expert,ch['weight']);fit,q=v.split(d,k,part)
            assert np.isfinite(p).all() and len(p)==len(q)
            if part=='final':
                assert len(p)==len(pos)
                for i,value in zip(pos,p):result[i][4]=repr(float(value))
            else:
                expected=next(r for r in ch['record']['confirmation'] if r['part']==part)
                actual=v.r83.score(d['metrics'][k],d['y'][q],p);np.testing.assert_allclose(actual,expected['candidate'],atol=1e-12,rtol=0)
                checks.append(dict(task=task,part=part,metric=d['metrics'][k],baseline=expected['baseline'],candidate=actual,gain=expected['gain']))
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();spans=field_spans(raw);pieces=[];cursor=0
    for i,row in enumerate(result[1:],1):
        lo,hi=spans[i][4];pieces.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    pieces.append(raw[cursor:]);payload=b''.join(pieces)
    folder=ROOT/'admet_submissions/v130_low_task_experts';folder.mkdir(exist_ok=True)
    file=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261027.csv'
    if file.exists():assert file.read_bytes()==payload
    else:file.write_bytes(payload)
    verification=verify(file,template);newspans=field_spans(payload);restored=[];cursor=0
    for i in range(1,len(result)):
        lo,hi=newspans[i][4];a,z=spans[i][4];restored.extend((payload[cursor:lo],raw[a:z]));cursor=hi
    restored.append(payload[cursor:]);assert b''.join(restored)==raw
    changed=Counter()
    for a,z in zip(original[1:],result[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[0] not in selected:assert a==z
        if a[4]!=z[4]:changed[a[0]]+=1
    assert set(changed)==set(selected)
    protected=[t for t in d['tasks'] if t not in selected]
    high=[r['task'] for r in json.loads((ROOT/'docs/platform_scorecard_v124_20261027.json').read_text())['tasks'] if r['score']>=51]
    assert not set(high)&set(changed)
    verification.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),unchanged_tasks=22-len(changed),protected_task_predictions_exact=True,high_score_tasks_unchanged=high,validation_replay='PASS')
    report=dict(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',version='v130',baseline='v124',baseline_official_score=51.115,
        baseline_sha256=v.ANCHOR_SHA,file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),selected=selected,
        validation_replay=checks,verification=verification,independent_model_verification_sha256=b.sha(ROOT/'admet_model_outputs_v129/independent_verification.json'),
        protected_tasks=protected,official_score=None,target_52_achieved=False,test_labels_used=False,external_data_added=False,
        automatically_uploaded=False,limitations='Public labels repeatedly reused. High score tasks unchanged exactly; no guarantee changed low task improves official score. Most baseline upstream OOF approximate; CYP3A4 is matched exact v67 target fit/query.')
    b.save(OUT/'evidence.json',report);b.save(OUT/'verification.json',verification);b.save(folder/'model_evidence.json',report);b.save(file.with_suffix('.verification.json'),verification)
    print(json.dumps(dict(status=report['status'],file=report['file'],sha256=report['sha256'],changed_tasks=dict(changed))),flush=True)
if __name__=='__main__':main()
