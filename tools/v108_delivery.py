"""Separate CPU forward replay, exact-template delivery, and v108 archive."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import argparse,csv,io,json,platform,sys,tarfile
from collections import Counter
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from scipy.special import expit
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v108_supervised_transfer as v
from verify_submission import verify
from repack_upload_template import field_spans
OUT=v.OUT

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))

def selected():
    c=read(OUT/'confirmation.json');f=read(OUT/'frozen_selection.json')
    assert c['status']=='COMPLETE' and c['signature']==f['signature']==v.sig()
    assert c['frozen_sha256']==v.data.sha(OUT/'frozen_selection.json')
    result=[r for r in c['records'] if r['accepted']]
    assert [r['task'] for r in result]==c['accepted']
    for r in result:
        assert any(x['task']==r['task'] for x in f['selected'])
        assert r['uncertainty']['percentile_95'][0]>0
        assert all(x.get('gain',1)>0 for x in r['confirmation'])
        assert all(x['transfer_gain']>=0 for x in r['confirmation'] if x['part'] in ('random','scaffold'))
    return result

def cpu_forward(w,fp,desc,k):
    def linear(x,p):return F.linear(x,w[p+'.weight'],w[p+'.bias'])
    def block(x,p):
        x=linear(x,p+'.0');x=F.layer_norm(x,(x.shape[-1],),w[p+'.1.weight'],w[p+'.1.bias'],1e-5)
        return F.silu(x)
    h=torch.cat([block(fp[:,i*2048:(i+1)*2048],f'views.{i}') for i in range(6)]+[block(desc,'desc')],1)
    h=block(block(h,'trunk.0'),'trunk.1')
    return linear(block(h,f'heads.{k}.0'),f'heads.{k}.1').squeeze(1)

def replay(d,k,part,kind,seed):
    folder=OUT/'models'/d['tasks'][k]/part/kind/str(seed);z=v.cached(folder)
    assert z is not None
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    assert ck['signature']==v.sig() and ck['k']==k and ck['part']==part and ck['kind']==kind
    fit,q=v.split(d,part);fit=fit[d['tid'][fit]==k];q=q[d['tid'][q]==k]
    for key,ix in (('fit',fit),('query',q)):
        np.testing.assert_array_equal(z[key],ix);np.testing.assert_array_equal(ck[key],ix)
    assert np.all(d['split'][fit]<2) and not set(fit)&set(q)
    # Independently reconstruct fit-only moments and endpoint normalization.
    allfit=v.split(d,part)[0];mm=np.unique(d['mi'][allfit]);raw=d['views']['physchem'].astype(float)
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in raw[mm].T])
    x=np.where(np.isfinite(raw[mm]),raw[mm],med);mean=x.mean(0);std=x.std(0).clip(.05)
    for n,x in (('median',med),('mean',mean),('std',std)):np.testing.assert_array_equal(ck[n],x)
    desc=((np.where(np.isfinite(raw[d['mi'][q]]),raw[d['mi'][q]],med)-mean)/std).clip(-6,6).astype(np.float32)
    fp=np.sqrt(d['views']['multiview'][d['mi'][q],:12288]).astype(np.float32)
    values=[]
    with torch.inference_mode():
        for lo in range(0,len(q),213):
            values.append(cpu_forward(ck['weights'],torch.from_numpy(fp[lo:lo+213]),torch.from_numpy(desc[lo:lo+213]),k).numpy())
    p=np.concatenate(values).astype(float);np.testing.assert_allclose(p,z['raw'],atol=3e-5,rtol=3e-5)
    # A second batch shape guards accidental dependence on query composition.
    ids=np.unique([0,len(q)//2,len(q)-1])
    with torch.inference_mode():sub=cpu_forward(ck['weights'],torch.from_numpy(fp[ids]),torch.from_numpy(desc[ids]),k).numpy()
    np.testing.assert_allclose(sub,p[ids],atol=3e-5,rtol=3e-5)
    rec=dict(task=d['tasks'][k],part=part,kind=kind,seed=seed,status='PASS',rows=len(q),
             cpu_raw_max_difference=float(np.max(np.abs(p-z['raw']))),model_sha256=v.data.sha(folder/'model.pt'))
    v.emit(stage='cpu_replay',**rec);return rec

def assemble(d):
    choices=selected()
    if not choices:
        report=dict(status='REJECTED_NO_SUPPORTED_CANDIDATE',version='v108',baseline='v101',official_score=None,accepted=[],new_submission_created=False)
        v.data.save(OUT/'release.json',report);v.emit(**report);return
    tasks=[r['task'] for r in choices]
    v.run_part(d,'final',tasks,kinds=('transfer',))
    checks=[]
    for r in choices:
        k=d['tasks'].index(r['task'])
        for part in ('fold0','fold1','fold2','valid','random','scaffold','final'):
            for kind in (('transfer',) if part=='final' else ('transfer','scratch')):
                for seed in v.SEEDS:checks.append(replay(d,k,part,kind,seed))
    old=rows(v.ANCHOR);new=[r.copy() for r in old];changed={}
    for r in choices:
        k=d['tasks'].index(r['task']);a=v.design(d,'final');p=v.predictions(d,k,'final','transfer',a)
        base=np.array([float(x[4]) for x in old[1:] if x[0]==r['task']]);p=v.mixed(r['metric'],base,p)
        ix=[i for i,x in enumerate(old) if i and x[0]==r['task']]
        assert len(ix)==len(p) and np.isfinite(p).all()
        for i,x in zip(ix,p):new[i][4]=repr(float(x))
        changed[r['task']]=sum(new[i][4]!=old[i][4] for i in ix)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();sp=field_spans(raw);chunks=[];cursor=0
    for i,row in enumerate(new[1:],1):
        lo,hi=sp[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks)
    file=ROOT/'admet_submissions/v108_supervised_transfer/我从山北来带着兰花草-小分子药物预测技术创新赛-20261026.csv'
    file.parent.mkdir(parents=True,exist_ok=True)
    if file.exists():assert file.read_bytes()==output
    else:file.write_bytes(output)
    actual=rows(file);observed=Counter()
    for a,z in zip(old[1:],actual[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[4]!=z[4]:observed[z[0]]+=1
        if z[0] not in tasks:assert a==z
    assert dict(observed)==changed
    ss=field_spans(output);back=[];cursor=0
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];x,z=sp[i][4];back.extend((output[cursor:lo],raw[x:z]));cursor=hi
    back.append(output[cursor:]);assert b''.join(back)==raw
    fmt=verify(file,template);fmt.update(nonprediction_bytes_exact=True,independent_cpu_forward='PASS')
    v.data.save(file.with_suffix('.verification.json'),fmt)
    v.data.save(OUT/'model_verification.json',dict(status='PASS',signature=v.sig(),checks=checks,submission=fmt))
    report=dict(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',version='v108',baseline='v101',anchor_sha256=v.ANCHOR_SHA,
        baseline_official_score_user_reported=50.932,file=file.relative_to(ROOT).as_posix(),sha256=v.data.sha(file),
        selected=choices,changed_tasks=changed,unchanged_tasks=22-len(changed),verification=fmt,official_score=None,
        protocol_sha256=v.data.sha(OUT/'protocol.json'),test_labels_used=False,new_external_data=False,new_pretrained_weights=False,
        automatically_uploaded=False,limitations=read(OUT/'protocol.json')['limitations'])
    v.data.save(OUT/'release.json',report);v.emit(stage='release',**report)

def package():
    rel=read(OUT/'release.json');assert rel['status'] in ('VERIFIED_CANDIDATE_AWAITING_PLATFORM','REJECTED_NO_SUPPORTED_CANDIDATE')
    if rel['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM':assert read(OUT/'model_verification.json')['status']=='PASS'
    import scipy,sklearn
    v.data.save(OUT/'environment.json',dict(python=sys.version,platform=platform.platform(),gpu=torch.cuda.get_device_name(),
        numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    paths=[OUT,ROOT/'tools/v108_supervised_transfer.py',ROOT/'tools/v108_delivery.py',ROOT/'tools/v108_part_worker.py',ROOT/'tools/run_v108_pipeline.py',ROOT/'docs/ITERATION_V108_20261026.md']
    paths.extend(ROOT.glob('v108*.log'))
    if 'file' in rel:paths.extend([ROOT/rel['file'],(ROOT/rel['file']).with_suffix('.verification.json')])
    files={}
    for path in paths:
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc':files[p.relative_to(ROOT).as_posix()]=p
    prefix='v108_supervised_transfer_20261026';folder=ROOT/'releases';archive=folder/(prefix+'.tar.gz');mp=folder/(prefix+'.manifest.json')
    assert not archive.exists(),'Do not overwrite archive'
    dependencies=['v101_multi_route_20261026']
    deps=[dict(path='releases/'+x+'.tar.gz',sha256=v.data.sha(folder/(x+'.tar.gz'))) for x in dependencies]
    v.data.save(mp,dict(version='v108',files=[dict(path=n,bytes=p.stat().st_size,sha256=v.data.sha(p)) for n,p in sorted(files.items())],
        dependencies=deps,test_labels_included=False,credentials_included=False))
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    record=dict(path=archive.relative_to(ROOT).as_posix(),sha256=v.data.sha(archive),bytes=archive.stat().st_size,files=len(files),version='v108',official_score=None)
    v.data.save(folder/(prefix+'.archive.json'),record);v.emit(stage='archive',**record)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['assemble','package']);args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    if args.stage=='assemble':assemble(v.load())
    else:package()
