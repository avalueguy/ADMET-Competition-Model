"""Audit rejected v89 checkpoint replay and globally excluded scaffold rows."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import json
import sys
import tarfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v89_joint_residual as v


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    d=v.load();records=[]
    for part in ('fold0','fold1'):
        q=np.flatnonzero((d['split']==0)&(d['legacy_folds']==int(part[-1])))
        fit=np.flatnonzero((d['split']==0)&~np.isin(d['groups'],d['groups'][q]))
        assert not set(d['groups'][fit])&set(d['groups'][q]);assert not set(d['canonical'][fit])&set(d['canonical'][q])
        unique=np.unique(d['mi'][fit]);raw=d['views']['physchem'].astype(float)
        median=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in raw[unique].T])
        full=np.where(np.isfinite(raw),raw,median);mean=full[unique].mean(0);std=full[unique].std(0).clip(.05)
        desc=torch.from_numpy(((full-mean)/std).clip(-5,5).astype(np.float32)).cuda()
        fp=torch.from_numpy(np.sqrt(d['views']['multiview'][:,:-435]).astype(np.float32)).cuda()
        expected_offset=np.zeros(len(q),np.float32);expected_states=[]
        for k,metric in enumerate(d['metrics']):
            ix=fit[d['tid'][fit]==k];active=d['tid'][q]==k;qq=q[active]
            base=v.r83.raw_proxy(d,ix);qp=v.r83.raw_proxy(d,qq);y=d['y'][ix]
            if metric in ('AUROC','AUPRC'):
                z,qz=v.r83.ranklog(base),v.r83.ranklog(qp);beta=v.r83.calibrate(y,z)
                expected_offset[active]=beta[0]+beta[1]*qz;state=dict(metric=metric,beta=beta.tolist(),scale=1.,center=0.)
            elif metric=='MAE':
                scale=max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-3)
                expected_offset[active]=qp;state=dict(metric=metric,scale=scale,center=float(np.median(y)))
            else:
                z,qz=v.r83.ranknorm(base),v.r83.ranknorm(qp);xx=np.c_[np.ones(len(z)),z]
                beta=np.linalg.solve(xx.T@xx+np.diag([1e-8,1.]),xx.T@v.r83.ranknorm(y));beta[1]=max(beta[1],.01)
                expected_offset[active]=beta[0]+beta[1]*qz;state=dict(metric=metric,beta=beta.tolist(),scale=1.,center=0.)
            expected_states.append(state)
        mols=np.unique(d['mi'][q]);pos=np.searchsorted(mols,d['mi'][q])
        for seed in v.SEEDS:
            folder=v.OUT/'models'/part/str(seed);meta=json.loads((folder/'metadata.json').read_text());cache=np.load(folder/'predictions.npz')
            assert meta['signature']==str(cache['signature'])==v.signature()
            assert v.b.sha(folder/'predictions.npz')==meta['predictions_sha256']
            np.testing.assert_array_equal(cache['fit'],fit);np.testing.assert_array_equal(cache['query'],q)
            np.testing.assert_array_equal(cache['snapshots'],v.SNAPSHOTS);np.testing.assert_array_equal(cache['qoffset'],expected_offset)
            for j,step in enumerate(v.SNAPSHOTS):
                path=folder/f'checkpoint_{step}.pt';assert v.b.sha(path)==meta['checkpoint_hashes'][path.name]
                ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['signature']==v.signature()
                assert ck['seed']==seed and ck['step']==step and ck['tasks']==len(d['tasks'])
                np.testing.assert_array_equal(ck['fit'],fit);np.testing.assert_array_equal(ck['query'],q)
                for key,expected in [('median',median),('mean',mean),('std',std),('qoffset',expected_offset)]:
                    np.testing.assert_allclose(ck[key],expected,atol=1e-10,rtol=1e-10)
                for actual,expected in zip(ck['states'],expected_states):
                    assert actual.keys()==expected.keys()
                    for key,value in expected.items():
                        if isinstance(value,str):assert actual[key]==value
                        else:np.testing.assert_allclose(actual[key],value,atol=1e-10,rtol=1e-10)
                net=v.Net(ck['fp_dim'],ck['desc_dim'],ck['tasks']).cuda().eval();net.load_state_dict(ck['state'])
                predictions=[]
                with torch.inference_mode():
                    # Batch 347 differs from the saved prediction batch 512.
                    for start in range(0,len(mols),347):
                        mm=mols[start:start+347];predictions.append(net(fp[mm],desc[mm]).cpu().numpy())
                values=np.concatenate(predictions);delta=values[pos,d['tid'][q]].astype(float)
                np.testing.assert_allclose(delta,cache['deltas'][j],atol=2e-5,rtol=2e-5)
                records.append(dict(part=part,seed=seed,step=step,status='PASS',sha256=v.b.sha(path),
                    max_abs_difference=float(np.max(np.abs(delta-cache['deltas'][j])))))
                del net
        del fp,desc;torch.cuda.empty_cache()
    frozen=json.loads((v.OUT/'frozen_selection.json').read_text());conf=json.loads((v.OUT/'confirmation.json').read_text())
    assert frozen['selected']==conf['accepted']==[] and conf['status']=='COMPLETE'
    report=dict(status='PASS',models=len(records),records=records,no_csv=True,
        scope='Exact global train/query exclusions, fit-only structural preprocessing, base calibration, checkpoint hashes, state reload and alternate-batch inference.',
        limitations='No accepted task; no confirm/final model. Failure applies to this fixed shared residual network, not all multitask neural models.')
    v.b.save(v.OUT/'model_verification.json',report)
    print(json.dumps({k:val for k,val in report.items() if k!='records'}),flush=True)


def package():
    report=json.loads((v.OUT/'model_verification.json').read_text());assert report['status']=='PASS'
    files={}
    for path in (v.OUT,ROOT/'tools/v89_joint_residual.py',Path(__file__),ROOT/'docs/ITERATION_V89_20261025.md'):
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc':files[p.relative_to(ROOT).as_posix()]=p
    folder=ROOT/'releases';prefix='v89_rejected_joint_residual_20261025';manifest=folder/(prefix+'.manifest.json')
    v.b.save(manifest,dict(files=[dict(path=n,sha256=v.b.sha(p),bytes=p.stat().st_size) for n,p in sorted(files.items())],
        dependency_archives=['v86_metric_ranking_20261025.tar.gz','v88_context_kernel_20261025.tar.gz'],
        test_labels_included=False,credentials_included=False,no_csv=True))
    archive=folder/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(manifest,arcname=manifest.relative_to(ROOT).as_posix())
    out=dict(path=archive.relative_to(ROOT).as_posix(),sha256=v.b.sha(archive),bytes=archive.stat().st_size,files=len(files),no_csv=True)
    v.b.save(folder/(prefix+'.archive.json'),out);print(json.dumps(out),flush=True)


if __name__=='__main__':
    (package if len(sys.argv)>1 and sys.argv[1]=='package' else main)()
