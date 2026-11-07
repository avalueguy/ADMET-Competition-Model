"""Assemble confirmed endpoint updates once over v94; no extra fitting/tuning."""
from pathlib import Path
from collections import Counter
import csv,io,json,sys,tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v95_graph_specialists as graph
import v96_normalized_assays as normalized
import v95_delivery as gd
import v96_delivery as nd
import v98_shape_experts as shape
import v98_delivery as sd
from verify_submission import verify
from repack_upload_template import field_spans
OUT=ROOT/'admet_model_outputs_v99'
PROTOCOL=ROOT/'docs/V98_PROTOCOL_20261026.json'

def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))

def run():
    assert graph.ANCHOR==normalized.ANCHOR and graph.ANCHOR_SHA==normalized.ANCHOR_SHA
    d=shape.load();chosen={};dependencies=[]
    for name,v,delivery in [('v95',graph,gd),('v96',normalized,nd),('v98',shape,sd)]:
        audit=json.loads((v.OUT/'model_verification.json').read_text());assert audit['status']=='PASS'
        rel=json.loads((v.OUT/'release.json').read_text()) if delivery.selected(False) else None
        if rel:assert rel['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM' and rel['verification_sha256']==graph.b.sha(v.OUT/'model_verification.json')
        dependencies.append(dict(version=name,verification_sha256=graph.b.sha(v.OUT/'model_verification.json'),
            confirmation_sha256=graph.b.sha(v.OUT/'confirmation.json'),signature=v.sig()))
        for r in delivery.selected(False):
            if r['task'] not in chosen or r['objective']>chosen[r['task']]['record']['objective']:
                chosen[r['task']]=dict(version=name,record=r)
    assert chosen,'No accepted endpoint updates.'
    old=read(graph.ANCHOR);rows=[r.copy() for r in old];changed={};checks=[]
    for task,selection in chosen.items():
        v={'v95':graph,'v96':normalized,'v98':shape}[selection['version']];r=selection['record'];k=d['tasks'].index(task)
        for part in ('fold0','fold1','fold2','valid','final'):
            p=v.mix(r['metric'],v.base(d,k,part),v.expert(d,k,part,r['config']),r['weight'])
            if part=='final':
                pos=[i for i,row in enumerate(old) if i and row[0]==task]
                assert len(p)==len(pos) and np.isfinite(p).all()
                for i,x in zip(pos,p):rows[i][4]=repr(float(x))
                changed[task]=sum(rows[i][4]!=old[i][4] for i in pos)
            else:
                q=v.r83.split(d,k,part)[1];record=next(x for x in r['confirmation'] if x['part']==part)
                score=v.r83.score(r['metric'],d['y'][q],p);np.testing.assert_allclose(score,record['candidate'],atol=1e-12,rtol=1e-12)
                checks.append(dict(task=task,version=selection['version'],part=part,score=score,gain=record['gain']))
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();sp=field_spans(raw);chunks=[];cursor=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=sp[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks)
    file=ROOT/'admet_submissions/v99_graph_assay_shape/我从山北来带着兰花草-小分子药物预测技术创新赛-20261026.csv';file.parent.mkdir(parents=True,exist_ok=True)
    if file.exists():assert file.read_bytes()==output
    else:file.write_bytes(output)
    actual=read(file);observed=Counter()
    for a,z in zip(old[1:],actual[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[4]!=z[4]:observed[z[0]]+=1
        if z[0] not in chosen:assert a==z
    assert dict(observed)==changed
    ss=field_spans(output);back=[];cursor=0
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];a,z=sp[i][4];back.extend((output[cursor:lo],raw[a:z]));cursor=hi
    back.append(output[cursor:]);assert b''.join(back)==raw
    fmt=verify(file,template);fmt.update(nonprediction_bytes_exact=True,model_replay='PASS',changed_tasks=changed)
    graph.b.save(file.with_suffix('.verification.json'),fmt);OUT.mkdir(exist_ok=True)
    report=dict(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',version='v99',baseline='v94',anchor_sha256=graph.ANCHOR_SHA,
        baseline_official_score_user_reported=50.916,file=file.relative_to(ROOT).as_posix(),sha256=graph.b.sha(file),
        selected=chosen,changed_tasks=changed,unchanged_tasks=22-len(changed),verification=fmt,checks=checks,
        protocol_sha256=graph.b.sha(PROTOCOL),dependencies=dependencies,official_score=None,test_labels_used=False,
        external_data_new_branch=False,new_pretrained_weights=False,automatically_uploaded=False)
    graph.b.save(OUT/'release.json',report)
    graph.b.save(OUT/'model_verification.json',dict(status='PASS',branches=dependencies,checks=checks,submission=fmt))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    graph.torch.set_num_threads(4);graph.torch.set_num_interop_threads(2)
    graph.torch.backends.cuda.matmul.allow_tf32=False;graph.torch.set_float32_matmul_precision('highest');run()


