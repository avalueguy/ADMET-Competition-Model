"""Independently reload broad-task winners, replay metrics and export exact CSV."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import csv,hashlib,io,json,sys
from collections import Counter
from pathlib import Path
import joblib,numpy as np,torch,xgboost as xgb
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v131_current_controls as v
import v133_broad_random_forest as forest
import v134_regression_kernels as kernel
import v135_broad_boosting as boost
import v136_neighbor_specialists as neighbor
import v137_pairwise_neural as neural
from repack_upload_template import field_spans
from verify_submission import verify
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v138'
MODULES={'v133':forest,'v134':kernel,'v135':boost,'v136':neighbor,'v137':neural}
def table(path):return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))
def reload(d,k,part,module,chosen):
    cfg=chosen['config'];task=d['tasks'][k];m=d['metrics'][k]
    if module is forest:
        fit,q,x,qx,source,median=forest.design(d,k,part,cfg);folder=forest.OUT/'models'/task/part/forest.key(cfg)
        mp=folder/'model.joblib';model=joblib.load(mp);p=model.predict_proba(qx)[:,1] if m in ('AUROC','AUPRC') else model.predict(qx);hashes={mp.name:b.sha(mp)}
        cache=np.load(folder/'predictions.npz');np.testing.assert_array_equal(median,cache['median'])
    elif module is kernel:
        fit,q,matrices=kernel.matrices(d,k,part);folder=kernel.OUT/'models'/task/part;mp=folder/(kernel.key(cfg)+'.joblib')
        cache=np.load(folder/(kernel.key(cfg)+'.npz'));model=joblib.load(mp);p=model.predict(matrices[cfg['view']][1])*float(cache['scale'])+float(cache['center']);hashes={mp.name:b.sha(mp)}
    elif module is neural:
        a=neural.design(d,k,part);fit,q=a['fit'],a['query'];folder=neural.OUT/'models'/task/part;cache=np.load(folder/'predictions.npz')
        fp=torch.as_tensor(a['fp'],device='cuda');desc=torch.as_tensor(a['desc'],device='cuda');members=[];hashes={}
        for seed in neural.SEEDS:
            mp=folder/f'model_{seed}.pt';state=torch.load(mp,weights_only=True);assert state['signature']==neural.sig()
            net=neural.Net(state['dim']).cuda();net.load_state_dict(state['weights']);members.append(neural.infer(net,a,fp,desc));hashes[mp.name]=b.sha(mp);del net
        raw=np.mean(members,axis=0);p=expit(raw) if m in ('AUROC','AUPRC') else raw*a['scale']+a['center'];del fp,desc;torch.cuda.empty_cache()
    else:
        fit,q,x,qx,source=(neighbor.design(d,k,part,cfg) if module is neighbor else v.features(d,k,part,cfg['view']))
        folder=module.OUT/'models'/task/part/module.key(cfg);cache=np.load(folder/'predictions.npz');members=[];hashes={}
        for seed in module.SEEDS:
            mp=folder/f'model_{seed}.ubj';model=xgb.Booster();model.load_model(mp);model.set_param({'device':'cpu'});members.append(model.predict(xgb.DMatrix(qx)));hashes[mp.name]=b.sha(mp)
        p=np.mean(members,axis=0).astype(float)*float(cache['scale'])+float(cache['center'])
    if task=='ppbr_az':p=p.clip(0,100)
    assert str(cache['signature'])==module.sig();np.testing.assert_array_equal(cache['fit'],fit);np.testing.assert_array_equal(cache['query'],q)
    np.testing.assert_allclose(p,cache['p'],atol=1e-10,rtol=0)
    return fit,q,p,hashes
def perturb(d,k,part,module,cfg):
    if module is neural:
        a=neural.design(d,k,part);q=a['query'];original=d['y'][q].copy();d['y'][q]=987654.;d.pop('_v96_assays',None);z=neural.design(d,k,part)
        d['y'][q]=original;d.pop('_v96_assays',None)
        for key in a:np.testing.assert_array_equal(a[key],z[key])
    else:
        before=v.features(d,k,part,'augmented');q=before[1];original=d['y'][q].copy();d['y'][q]=987654.;d.pop('_v96_assays',None);after=v.features(d,k,part,'augmented')
        d['y'][q]=original;d.pop('_v96_assays',None)
        for a,z in zip(before,after):np.testing.assert_array_equal(a,z)
    return True
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    d=v.load();selected={};experiments=[];nominees={}
    for name,module in MODULES.items():
        f=json.loads((module.OUT/'frozen_selection.json').read_text());c=json.loads((module.OUT/'confirmation.json').read_text())
        assert c['status']=='COMPLETE' and c['signature']==f['signature']==module.sig()
        experiments.append(dict(version=name,accepted=c['accepted'],signature=module.sig(),confirmation_sha256=b.sha(module.OUT/'confirmation.json')))
        for r in f['selected']:
            # Choose the development winner even when it later fails confirmation.
            # A rejected winner is not replaced by a lower-development route.
            if r['task'] not in nominees or r['objective']>nominees[r['task']]['record']['objective']:
                conf=next(z for z in c['records'] if z['task']==r['task'])
                assert all(conf[key]==r[key] for key in r)
                nominees[r['task']]=dict(version=name,record=conf)
    for task,choice in nominees.items():
        if choice['record']['accepted']:selected[task]=choice
    b.save(OUT/'route_winners.json',dict(status='FROZEN_BY_DEVELOPMENT_RULE',selection='Highest minimum development gain across frozen routes, regardless of confirmation; no fallback.',nominees=nominees,selected=selected))
    assert selected,'No confirmed broad-task development winners; no renamed baseline submission.'
    original=table(v.ANCHOR);result=[r.copy() for r in original];checks=[];rng=np.random.default_rng(1384026)
    for task,ch in selected.items():
        module=MODULES[ch['version']];r=ch['record'];k=d['tasks'].index(task);pos=[i for i,z in enumerate(original) if i and z[0]==task]
        for part in v.PARTS:
            fit,q,p,hashes=reload(d,k,part,module,r);a=v.baseline(d,k,part);candidate=v.mix(d['metrics'][k],a,p,r['weight']);assert np.isfinite(candidate).all()
            rec=dict(task=task,version=ch['version'],part=part,model_reload=True,model_hashes=hashes,fit=len(fit),query=len(q),query_label_perturbation=perturb(d,k,part,module,r['config']))
            if part=='final':
                assert len(pos)==len(candidate)
                for i,value in zip(pos,candidate):result[i][4]=repr(float(value))
            else:
                expected=next(z for z in r['confirmation'] if z['part']==part);actual=r83.score(d['metrics'][k],d['y'][q],candidate)
                np.testing.assert_allclose(actual,expected['candidate'],atol=1e-12,rtol=0);rec.update(expected)
                if part in ('fold2','valid'):
                    groups=np.unique(d['groups'][q]);deltas=[]
                    for _ in range(1000):
                        draw=rng.choice(groups,len(groups),replace=True);ix=np.concatenate([np.flatnonzero(d['groups'][q]==g) for g in draw])
                        if d['metrics'][k] in ('AUROC','AUPRC') and len(np.unique(d['y'][q][ix]))<2:continue
                        value=r83.gain(d['metrics'][k],d['y'][q][ix],a[ix],candidate[ix])
                        if np.isfinite(value):deltas.append(value)
                    rec.update(bootstrap_samples=len(deltas),scaffold_bootstrap_gain_ci95=np.quantile(deltas,[.025,.975]).tolist())
            checks.append(rec);print(json.dumps(dict(stage='verified',**rec)),flush=True)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();ss=field_spans(raw);pieces=[];cursor=0
    for i,row in enumerate(result[1:],1):
        lo,hi=ss[i][4];pieces.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    pieces.append(raw[cursor:]);payload=b''.join(pieces)
    folder=ROOT/'admet_submissions/v138_broad_task_models';folder.mkdir(parents=True,exist_ok=True)
    file=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261027.csv'
    if file.exists():assert file.read_bytes()==payload
    else:file.write_bytes(payload)
    fmt=verify(file,template);assert fmt['utf8_bom'] and fmt['crlf_only']
    outspans=field_spans(payload);restore=[];cursor=0
    for i in range(1,len(result)):
        lo,hi=outspans[i][4];a,z=ss[i][4];restore.extend((payload[cursor:lo],raw[a:z]));cursor=hi
    restore.append(payload[cursor:]);assert b''.join(restore)==raw
    changed=Counter()
    for a,z in zip(original[1:],result[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[0] not in selected:assert a==z
        if a[4]!=z[4]:changed[a[0]]+=1
    assert set(changed)==set(selected) and 'hia_hou' not in changed
    fmt.update(nonprediction_bytes_exact=True,protected_task_predictions_exact=True,changed_tasks=dict(changed),unchanged_tasks=22-len(changed))
    report=dict(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',version='v138',baseline='v130',baseline_official_score=51.146,baseline_sha256=v.ANCHOR_SHA,
        file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),selected=selected,route_nominees=nominees,experiments=experiments,checks=checks,verification=fmt,
        official_score=None,target_52_achieved=False,test_labels_used=False,external_data_added=False,automatically_uploaded=False,
        limitations='Historically reused development labels. Several baseline OOF controls approximate. Bootstrap does not remove selection bias. New task changes can regress officially; unmodified outputs exact.')
    b.save(OUT/'evidence.json',report);b.save(OUT/'independent_verification.json',dict(status='PASS',checks=checks,selected=selected,baseline=v.ANCHOR_SHA))
    b.save(folder/'model_evidence.json',report);b.save(file.with_suffix('.verification.json'),fmt)
    print(json.dumps(dict(status=report['status'],file=report['file'],sha256=report['sha256'],changed_tasks=dict(changed))),flush=True)
if __name__=='__main__':main()
