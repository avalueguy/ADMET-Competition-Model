"""Independent standard-library replay of the final combined submission."""
import csv,hashlib,io,json,math
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read(path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))

def rank(values):
    order=sorted(range(len(values)),key=values.__getitem__)
    result=[0.0]*len(values);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[start]]==values[order[end]]:end+=1
        for index in order[start:end]:result[index]=(start+end)/(2*len(values))
        start=end
    return result

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    dest=ROOT/'admet_submissions/v64';file=next(dest.glob('*.csv'))
    baseline_file=next((ROOT/'admet_submissions/v60').glob('*.csv'))
    raw=file.read_bytes();rows=read(file);baseline=read(baseline_file)
    template=read(ROOT/'admet_official/submission_template.csv')
    out=ROOT/'admet_model_outputs_v64'
    meta=json.loads((dest/'verification.json').read_text(encoding='utf-8'))
    release=json.loads((out/'release.json').read_text(encoding='utf-8'))
    replay=json.loads((out/'fusion_replay.json').read_text(encoding='utf-8'))
    assert sha(file)==meta['sha256']==release['sha256']
    assert sha(baseline_file)==meta['baseline_sha256']
    assert len(rows)==len(baseline)==len(template)==16380
    assert rows[0]==template[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
    assert raw.startswith(b'\xef\xbb\xbf') and b'\n' not in raw.replace(b'\r\n',b'')
    keys=set();changed=Counter();task_counts=Counter()
    for row,base,expected in zip(rows[1:],baseline[1:],template[1:]):
        assert len(row)==6 and row[:4]+row[5:]==expected[:4]+expected[5:]
        key=tuple(row[:2]);assert key not in keys;keys.add(key)
        value=float(row[4]);assert math.isfinite(value)
        if row[5] in ('AUROC','AUPRC'):assert 0<=value<=1
        if row[0] not in replay:assert row==base
        if row[4]!=base[4]:changed[row[0]]+=1
        task_counts[row[0]]+=1
    assert len(task_counts)==22
    checks=[]
    for task,record in replay.items():
        old=[float(row[4]) for row in baseline[1:] if row[0]==task]
        actual=[float(row[4]) for row in rows[1:] if row[0]==task]
        assert old==record['baseline'] and actual==record['result']
        branch=record['branch'];weight=record['selection']['weight']
        assert len(branch)==len(actual) and all(map(math.isfinite,branch))
        if record['metric']!='MAE':old,branch=rank(old),rank(branch)
        expected=[(1-weight)*a+weight*b for a,b in zip(old,branch)]
        if task=='ppbr_az':expected=[min(100.,max(0.,x)) for x in expected]
        error=max(abs(a-b) for a,b in zip(expected,actual));assert error<1e-12
        version=record['source_version'];source=ROOT/f'admet_model_outputs_v{version}'
        source_replay=json.loads((source/'fusion_replay.json').read_text(encoding='utf-8'))
        assert {key:value for key,value in record.items() if key!='source_version'}==source_replay[task]
        fresh=json.loads((source/'fresh.json').read_text(encoding='utf-8'))
        accepted=next(item for item in fresh['tasks'] if item['task']==task)
        assert fresh['status']=='COMPLETE' and accepted['pass_gate']
        model_check=json.loads((source/'model_replay.json').read_text(encoding='utf-8'))
        assert model_check['status']=='PASS'
        assert any(item['task']==task and item['model_reload_passed'] for item in model_check['tasks'])
        if version==61:
            extra=json.loads((source/'independent_chemical_gate.json').read_text(encoding='utf-8'))
            assert extra['status']=='COMPLETE'
            assert all(item['pass_gate'] for item in extra['tasks'] if item['task']==task)
        checks.append(dict(task=task,source_version=version,fusion_error=error,fresh_gate_passed=True,source_model_replay_passed=True))
    assert dict(changed)==meta['changed_tasks']
    audit=dict(status='PASS',sha256=sha(file),rows=16379,columns=6,tasks=22,
               bytes=len(raw),changed_tasks=dict(changed),changed_rows=sum(changed.values()),
               nonprediction_fields_exact=True,unselected_rows_exact=True,
               independent_fusion_replay=True,source_provenance_verified=True,
               task_checks=checks,encoding='UTF-8 BOM',line_endings='CRLF',
               official_score=None,automatically_uploaded=False,local_gpu_training=False)
    (dest/'local_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
