"""Matched scaffold confirmation for provisional candidates.

Historical tree, transfer and MTL recipes are retrained on exactly the same
folds. Source-task training records
with any held-out scaffold are excluded for strict transfer stress testing.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from admet_competition_train import fit_candidates, predict_model, featurize_smiles
from admet_features_v2 import features
from admet_transfer_upgrade import REL, transfer, heads
from admet_upgrade_v3 import metrics, dominates, train_heads, utility
from admet_v5_remote_train import fit_family, predict
from threadpoolctl import threadpool_limits


def train_legacy(recipe, raw_train, smiles_train, y, raw_hold, smiles_hold, task, tasks, cache, neural_prediction=None):
    k,metric=recipe['kind'],recipe['metric']
    xb,_,_=featurize_smiles(smiles_train)
    xh,_,_=featurize_smiles(smiles_hold)
    old=np.zeros(len(smiles_hold))
    for name,w in recipe['baseline_weights'].items():
        if w<=1e-10: continue
        model=fit_candidates(k,y)[name]
        if hasattr(model,'n_jobs'): model.set_params(n_jobs=6)
        model.fit(xb,y);old+=w*predict_model(model,xh,k)
    a=recipe['v3_alpha']
    if a:
        pp,_=train_heads(raw_train,y,k,metric,{'hold':raw_hold})
        chosen=recipe['v3_head']
        p=np.mean(list(pp['hold'].values()),axis=0) if chosen=='equal' else pp['hold'][chosen]
        old=(1-a)*old+a*p
    if recipe['neural_weight']:
        assert neural_prediction is not None
        old=(1-recipe['neural_weight'])*old+recipe['neural_weight']*neural_prediction
    a=recipe['v4_alpha']
    if a:
        rt,tt=transfer(task,smiles_train,tasks,cache)
        rh,th=transfer(task,smiles_hold,tasks,cache)
        xt=np.nan_to_num(np.hstack([rt,tt]),nan=0.,posinf=0.,neginf=0.)
        xh=np.nan_to_num(np.hstack([rh,th]),nan=0.,posinf=0.,neginf=0.)
        selected=recipe['v4_head']; models=heads(k)
        chosen=list(models) if selected=='equal' else [selected]
        ps=[]
        for name in chosen:
            m=models[name]; m.fit(xt[:,4096:] if name=='linear' else xt,y)
            ps.append(predict_model(m,xh[:,4096:] if name=='linear' else xh,k))
        old=(1-a)*old+a*np.mean(ps,axis=0)
    return old


def matched_task(task, result, protocol, recipes, tasks, out):
    recipe=recipes[task];dest=out/task;dest.mkdir(parents=True,exist_ok=True)
    tr=tasks[task]['train']; n=len(tr);smiles=tr.Drug.tolist();y=tr.Y.to_numpy(float)
    raw,canon,scaffolds=features(smiles,out/'features')
    with np.load(Path('admet_pretrained/embeddings')/(task+'.npz')) as z: emb=z['x'][:n]
    fold_df=pd.read_csv(Path('admet_model_outputs_v5')/task/'folds.csv',keep_default_na=False)
    assert fold_df.canonical.tolist()==canon and fold_df.scaffold.tolist()==scaffolds
    pred=pd.read_csv(Path('admet_model_outputs_v5')/task/(result['chosen']+'_oof.csv'))
    assert pred.row.tolist()==list(range(n)) and np.allclose(pred.Y_true,y)
    base=np.zeros(n);rows=[]; audits=[]
    for fold in range(protocol['cv_folds']):
        hold=np.flatnonzero(fold_df.fold.to_numpy()==fold); fit=np.flatnonzero(fold_df.fold.to_numpy()!=fold)
        filtered=copy.copy(tasks);forbidden=set(np.asarray(scaffolds)[hold])
        for source in REL.get(task,[]):
            if source not in tasks:continue
            df=tasks[source]['train'];_,_,sg=features(df.Drug.tolist(),out/'features')
            mask=np.array([s not in forbidden for s in sg])
            assert not set(np.asarray(sg)[mask])&forbidden
            filtered[source]={'train':df.loc[mask].reset_index(drop=True)}
            audits.append(dict(task=task,fold=fold,source=source,removed=int((~mask).sum()),remaining=int(mask.sum())))
        npred=None
        if recipe['neural_weight']:
            from admet_v5_legacy_mtl import fit_predict
            npred=fit_predict(task,fold,forbidden,[smiles[i] for i in hold],tasks,out/'legacy_mtl')
        base[hold]=train_legacy(recipe,raw[fit],[smiles[i] for i in fit],y[fit],
                                raw[hold],[smiles[i] for i in hold],task,filtered,out/'features',npred)
        blend=(1-result['alpha'])*base[hold]+result['alpha']*pred.Y_pred.to_numpy(float)[hold]
        old=metrics(recipe['kind'],y[hold],base[hold]);new=metrics(recipe['kind'],y[hold],blend)
        # Strict double-metric gate in every predeclared fold, no alpha tuning.
        ok=dominates(recipe['kind'],old,new)
        rows.append(dict(fold=fold,passed=ok,**{'before_'+k:v for k,v in old.items()},**{'after_'+k:v for k,v in new.items()}))
        print(json.dumps(dict(task=task,stage='matched_confirmation',**rows[-1])),flush=True)
    accepted=bool(all(r['passed'] for r in rows))
    status=dict(task=task,accepted=accepted,folds=rows,valid=result['provisional_valid_pass'],
                recipe='historical tree/transfer/MTL with source scaffold exclusion',
                limitation='CV reused for candidate selection; no independent blind score guarantee')
    pd.DataFrame(audits).to_csv(dest/'source_exclusion.csv',index=False)
    pd.DataFrame(dict(row=np.arange(n),Y_true=y,baseline=base,candidate=pred.Y_pred)).to_csv(dest/'matched_oof.csv',index=False)
    (dest/'confirmation.json').write_text(json.dumps(status,indent=2),encoding='utf8')
    return status


def main():
    if os.name == 'nt' or not Path('/root/autodl-tmp/admet_20260925').is_dir():
        raise RuntimeError('Confirmation training is restricted to the requested remote workspace')
    ap=argparse.ArgumentParser();ap.add_argument('--tasks',nargs='*');a=ap.parse_args()
    protocol=json.loads(Path('admet_v5_protocol.json').read_text())
    recipes=json.loads(Path('admet_legacy_recipes.json').read_text())
    root=Path('admet_competition_data/data_benchmark')
    tasks={d.name:{s:pd.read_csv(d/(s+'.csv')) for s in ['train','valid','test']} for d in root.iterdir()}
    out=Path('admet_v5_confirmation');out.mkdir(exist_ok=True)
    with threadpool_limits(limits=6):
        for task in a.tasks or sorted(tasks):
            p=Path('admet_model_outputs_v5')/task/'result.json'
            if not p.exists():continue
            result=json.loads(p.read_text())
            if not result['provisional_valid_pass']:continue
            if (out/task/'confirmation.json').exists():
                prior=json.loads((out/task/'confirmation.json').read_text())
                if not prior.get('reason','').startswith('Full historical MTL'):continue
                (out/task/'prior_incomplete_confirmation.json').write_text(json.dumps(prior,indent=2))
            print(json.dumps(matched_task(task,result,protocol,recipes,tasks,out)),flush=True)


if __name__=='__main__':main()
