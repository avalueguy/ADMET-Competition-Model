"""Independent saved-network replay and strict v175 task assembly."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import csv,io,json,sys,time
from pathlib import Path
from collections import Counter
import joblib,numpy as np,torch
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v175_shared_graph_hybrid as m
import v148_experiment_runner as runner
from repack_upload_template import field_spans
v,b,r83=m.v,m.b,m.r83
OUT=ROOT/'admet_model_outputs_v176'

def same(a,z):
    if isinstance(a,dict):
        assert set(a)==set(z)
        for k in a:same(a[k],z[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(z)
        for x,y in zip(a,z):same(x,y)
    else:np.testing.assert_array_equal(a,z)

def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=True);cp=m.OUT/'confirmation.json'
    for _ in range(720):
        if cp.exists() and json.loads(cp.read_text())['status']=='COMPLETE':break
        time.sleep(15)
    else:raise TimeoutError('Training incomplete; retain all evidence')
    conf=json.loads(cp.read_text());frozen=json.loads((m.OUT/'frozen_selection.json').read_text())
    assert conf['signature']==frozen['signature']==m.sig()
    selected={r['task']:r for r in conf['records'] if r['accepted']}
    if not selected:
        b.save(OUT/'no_candidate.json',dict(status='NO_CONFIRMED_NEW_MODEL',training_version='v175',baseline='v145',official_baseline_score=51.166,submission_created=False,confirmation_sha256=b.sha(cp)))
        print('NO_CONFIRMED_NEW_MODEL',flush=True);return
    d=v.load();graphs=m.graph.build_graphs(d);c=m.CONFIGS[0];checks=[];replayed={}
    for part in v.PARTS:
        a=m.design(d,part);folder=m.OUT/'models'/part/m.key(c);cache=np.load(folder/'predictions.npz');assert str(cache['signature'])==m.sig()
        saved=joblib.load(folder/'design.joblib');same(saved,{name:value for name,value in a.items() if name not in ('fp','desc')});same(cache['source'],a['source'])
        allq=np.unique(np.concatenate(list(a['queries'].values())));ys=d['y'][allq].copy()
        try:
            d['y'][allq]=987654.;same(a,m.design(d,part))
        finally:d['y'][allq]=ys
        fp=torch.as_tensor(a['fp'],device='cuda');desc=torch.as_tensor(a['desc'],device='cuda');raws={d['tasks'].index(t):[] for t in selected};hashes={}
        for seed in m.SEEDS:
            mp=folder/f'model_{seed}.pt';state=torch.load(mp,weights_only=True);assert state['signature']==m.sig() and state['config']==c
            net=m.Net(state['ad'],state['ed'],state['dd'],state['tasks'],c).cuda();net.load_state_dict(state['state']);hashes[mp.name]=b.sha(mp)
            for k in raws:raws[k].append(m.infer(net,d,a,graphs,fp,desc,k,'cuda'))
            del net
        for k,raw in raws.items():
            task=d['tasks'][k];state=a['states'][k];p=np.mean(raw,axis=0);p=expit(p) if d['metrics'][k] in ('AUROC','AUPRC') else p*state['scale']+state['center']
            if task=='ppbr_az':p=p.clip(0,100)
            same(p,cache[f'p{k}']);same(a['queries'][k],cache[f'query{k}']);replayed[(task,part)]=p
            checks.append(dict(task=task,part=part,model_reload=True,query_label_perturbation_invariant=True,global_scaffold_exclusion=True,model_hashes=hashes))
        del fp,desc;torch.cuda.empty_cache()
        print(json.dumps(dict(stage='independent_replay',part=part,tasks=list(selected))),flush=True)
    original=read(v.ANCHOR);rows=[r.copy() for r in original]
    for task,r in selected.items():
        k=d['tasks'].index(task);ch=next(x for x in frozen['selected'] if x['task']==task)
        assert runner.confirm(d,k,m.predict,ch)==r
        for part in ('fold0','fold1','fold2','valid'):
            q=v.split(d,k,part)[1];p=v.mix(d['metrics'][k],v.baseline(d,k,part),replayed[(task,part)],r['weight']);expected=next(z for z in r['confirmation'] if z['part']==part)
            np.testing.assert_allclose(r83.score(d['metrics'][k],d['y'][q],p),expected['candidate'],atol=1e-12,rtol=0)
        p=v.mix(d['metrics'][k],v.baseline(d,k,'final'),replayed[(task,'final')],r['weight']);positions=[i for i,z in enumerate(rows) if i and z[0]==task];assert len(positions)==len(p)
        for i,value in zip(positions,p):rows[i][4]=repr(float(value))
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();spans=field_spans(raw);pieces=[];cursor=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=spans[i][4];pieces.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    pieces.append(raw[cursor:]);payload=b''.join(pieces)
    folder=ROOT/'admet_submissions/v176_shared_graph_hybrid';folder.mkdir(exist_ok=True);file=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261027.csv'
    if file.exists():assert file.read_bytes()==payload
    else:file.write_bytes(payload)
    check=read(file);reference=read(template);assert len(check)==len(rows)==len(reference)==16380 and check==rows
    changed=Counter();keys=set()
    for a,z,t in zip(original[1:],check[1:],reference[1:]):
        assert z[:4]+z[5:]==t[:4]+t[5:];key=tuple(z[:2]);assert key not in keys;keys.add(key);value=float(z[4]);assert np.isfinite(value)
        if z[5] in ('AUROC','AUPRC'):assert 0<=value<=1
        if a[0] not in selected:assert a==z
        if a[4]!=z[4]:changed[a[0]]+=1
    assert set(changed)==set(selected) and 'hia_hou' not in changed
    outspans=field_spans(payload);pieces=[];cursor=0
    for i in range(1,len(check)):
        lo,hi=outspans[i][4];a,z=spans[i][4];pieces.extend((payload[cursor:lo],raw[a:z]));cursor=hi
    pieces.append(payload[cursor:]);assert b''.join(pieces)==raw
    fmt=dict(status='PASS',sha256=b.sha(file),rows=16379,columns=6,tasks=22,nonprediction_bytes_exact=True,original_drug_id_strings_exact=True,finite_predictions=True,whole_task_routing=True,independent_final_fusion_replay=True,model_reload=True,query_label_perturbation_invariant=True,changed_tasks=dict(changed),unchanged_tasks=22-len(changed),utf8_bom=payload.startswith(b'\xef\xbb\xbf'),crlf_only=b'\n' not in payload.replace(b'\r\n',b''))
    report=dict(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',version='v176',training_version='v175',baseline='v145',baseline_official_score=51.166,baseline_sha256=v.ANCHOR_SHA,file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),selected=selected,checks=checks,verification=fmt,official_score=None,automatically_uploaded=False,test_labels_used=False,new_external_data=False,limitations='Public labels reused; historical controls partly approximate. Independent replay verifies implementation; it cannot guarantee official gain.')
    b.save(OUT/'evidence.json',report);b.save(OUT/'independent_verification.json',dict(status='PASS',checks=checks));b.save(folder/'model_evidence.json',report);b.save(file.with_suffix('.verification.json'),fmt)
    print(json.dumps(dict(status=report['status'],changed_tasks=dict(changed),sha256=report['sha256'])),flush=True)
if __name__=='__main__':main()
