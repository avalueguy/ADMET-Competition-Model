"""Repeat-split robustness check, refit, replay and official-template export."""
import argparse
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from admet_v6_runner import (OUT, remote_only, load_task, all_tasks, legacy_hold, fit_candidate,
                             predict_candidate, metrics, kind, deltas, secondary_ok, save, emit, sha)
from admet_submission import build


def confirm(task,protocol):
    dest=OUT/task;path=dest/'decision.json';decision=json.loads(path.read_text())
    if not decision.get('valid_pass') or 'repeat_confirmation' in decision:return decision
    sel=decision['selection'];data=load_task(task);n=len(data['frames']['train'])
    y=data['frames']['train'].Y.to_numpy(float);metric=sel['metric'];kk=kind(metric)
    it,iv=next(GroupShuffleSplit(n_splits=1,test_size=protocol['confirmation_fraction'],
                   random_state=protocol['confirmation_seed']).split(np.arange(n),y,data['groups'][:n]))
    assert not set(data['groups'][it])&set(data['groups'][iv])
    assert not set(data['canonical'][it])&set(data['canonical'][iv])
    if kk=='classification' and (len(np.unique(y[iv]))<2 or len(np.unique(y[it]))<2):
        decision.update(accepted=False,reason='repeat split class coverage failed')
        save(path,decision);return decision
    base=legacy_hold(task,data,it,iv,all_tasks(),'repeat_692509')
    route=json.loads((Path('admet_release_v5')/task/'selection.json').read_text())
    if route['accepted']:
        oldsel=json.loads((Path('admet_model_outputs_v5')/task/'result.json').read_text())
        model=fit_candidate('v5_'+oldsel['chosen'],data,it,y[it],metric,protocol)
        base=.75*base+.25*predict_candidate(model,data,iv)
    model=fit_candidate(sel['family'],data,it,y[it],metric,protocol)
    cand=predict_candidate(model,data,iv);q=(1-sel['alpha'])*base+sel['alpha']*cand
    old=metrics(kk,y[iv],base);new=metrics(kk,y[iv],q)
    tol=.01 if metric=='MAE' else .005
    passed=deltas(metric,old,new)>=-tol and secondary_ok(metric,old,new)
    decision.update(accepted=bool(passed),repeat_confirmation=dict(before=old,after=new,passed=bool(passed),
                    fit_rows=len(it),holdout_rows=len(iv),seed=protocol['confirmation_seed']),
                    reason='passed observed OOF, valid and repeat-split gates' if passed else 'repeat scaffold gate failed')
    pd.DataFrame(dict(row=iv,Y_true=y[iv],baseline=base,candidate=cand,blend=q)).to_csv(dest/'repeat_confirmation.csv',index=False)
    save(path,decision);emit(stage='repeat',**decision)
    return decision


def publish(tasks,protocol):
    out=Path('admet_release_v6');out.mkdir(exist_ok=True);routes=[]
    names=sorted(p.name for p in Path('admet_competition_data/data_benchmark').iterdir() if p.is_dir())
    for task in names:
        prior=Path('admet_release_v5')/task;dest=out/task;dest.mkdir(exist_ok=True)
        decision_path=OUT/task/'decision.json'
        decision=json.loads(decision_path.read_text()) if decision_path.exists() else dict(accepted=False,reason='not evaluated')
        accepted=decision['accepted'] and task in tasks
        if accepted:
            data=load_task(task);n=len(data['frames']['train'])+len(data['frames']['valid'])
            y=np.r_[data['frames']['train'].Y,data['frames']['valid'].Y];sel=decision['selection']
            cached=dest/'model.joblib';prior_selection=dest/'selection.json'
            reuse=False
            if cached.exists() and prior_selection.exists():
                previous=json.loads(prior_selection.read_text())
                reuse=(previous.get('selection',{}).get('family')==sel['family'] and
                       previous.get('selection',{}).get('protocol_sha256')==sel['protocol_sha256'] and
                       previous.get('model_sha256')==sha(cached))
            if reuse:
                model=joblib.load(cached)
                if sel['family'].startswith('kernel'):
                    from admet_v6_features import mixed_kernel
                    data['kernel']=mixed_kernel(data)
            else:model=fit_candidate(sel['family'],data,np.arange(n),y,sel['metric'],protocol)
            itest=np.arange(n,len(data['smiles']));p=predict_candidate(model,data,itest)
            joblib.dump(model,dest/'model.joblib',compress=3)
            p2=predict_candidate(joblib.load(dest/'model.joblib'),data,itest)
            np.testing.assert_allclose(p,p2,atol=2e-6,rtol=2e-6)
            old=pd.read_csv(prior/f'submission_{task}.csv',dtype={'Drug_ID':str},keep_default_na=False)
            weight=sel['alpha']
            if decision.get('domain_guard',{}).get('passed'):
                from admet_v6_domain_guard import nearest
                weight=np.where(nearest(data['raw'],np.arange(n),itest)<.3,0.,weight)
            final=(1-weight)*old.Y.to_numpy(float)+weight*p
            if sel['kind']=='classification':assert ((final>=0)&(final<=1)).all()
            assert np.isfinite(final).all()
            pd.DataFrame(dict(Drug_ID=old.Drug_ID,Y=final)).to_csv(dest/f'submission_{task}.csv',index=False)
            pd.DataFrame(dict(Drug_ID=old.Drug_ID,Y=p)).to_csv(dest/'candidate_test.csv',index=False)
            comp=pd.read_csv(OUT/task/'validation_comparison.csv')
            pd.DataFrame(dict(Y_true=comp.Y_true,Y_pred=comp.blend)).to_csv(dest/'validation_predictions.csv',index=False)
            decision.update(model_sha256=sha(dest/'model.joblib'),feature_sha256=data['feature_sha256'],replay_pass=True)
        else:
            for fn in [f'submission_{task}.csv','validation_predictions.csv']:shutil.copy2(prior/fn,dest/fn)
            assert (dest/f'submission_{task}.csv').read_bytes()==(prior/f'submission_{task}.csv').read_bytes()
        decision.update(task=task,accepted=bool(accepted),base_sha256=sha(prior/f'submission_{task}.csv'),
                        prediction_sha256=sha(dest/f'submission_{task}.csv'))
        save(dest/'selection.json',decision);routes.append(decision)
        emit(stage='release',task=task,accepted=bool(accepted))
    save(out/'routing.json',routes)
    manifest=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
              for p in sorted(Path('.').glob('admet_v6*')) if p.is_file()]
    save(out/'source_manifest.json',manifest)
    if any(r['accepted'] for r in routes):
        path=Path('admet_submissions/admet-v6-20260925.csv');build(out,path)
        df=pd.read_csv(path,dtype={'drug_id':str},keep_default_na=False)
        upload=Path('admet_submissions/v6/我从山北来带着兰花草-小分子药物预测技术创新赛-20260925.csv')
        upload.parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(upload,index=False,encoding='utf-8-sig',lineterminator='\r\n')
        read=pd.read_csv(upload,dtype={'drug_id':str},keep_default_na=False)
        assert len(read)==16379 and list(read.columns)==['task','sample_idx','drug_id','smiles','y_pred','metric']
        assert read.drop(columns='y_pred').equals(df.drop(columns='y_pred'))
        assert upload.read_bytes().count(b'task,sample_idx,drug_id,smiles,y_pred,metric')==1
        save(out/'verification.json',dict(status='VERIFIED',rows=len(read),tasks=read.task.nunique(),
             accepted=[r['task'] for r in routes if r['accepted']],submission=str(upload),sha256=sha(upload),
             protocol_sha256=sha('admet_v6_protocol.json'),platform_score=None,
             limitation='Validation reused; hidden-set improvements and target score not guaranteed'))
    else:
        save(out/'verification.json',dict(status='NO_PROMOTION',platform_baseline=48.98128))


def main():
    remote_only();ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['confirm','publish']);ap.add_argument('--tasks',nargs='*')
    args=ap.parse_args();protocol=json.loads(Path('admet_v6_protocol.json').read_text())
    tasks=args.tasks or sorted(p.parent.name for p in OUT.glob('*/decision.json'))
    with threadpool_limits(limits=6):
        if args.stage=='confirm':
            for task in tasks:confirm(task,protocol)
        else:publish(tasks,protocol)


if __name__=='__main__':main()
