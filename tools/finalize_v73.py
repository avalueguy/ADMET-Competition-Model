"""Fit the frozen v73 endpoint set on train+valid and export from v72."""
from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v73_experts as e
import admet_v73_conditional as c
from repack_upload_template import field_spans
from verify_submission import verify

SEEDS=(732026,732027)
STRUCTURE={
 'ames':('multiview',.25),
 'clearance_microsome_az':('multiview',.10),
 'cyp2d6_veith':('multiview',.25),
 'dili':('multiview',.25),
}
CONDITIONAL={'cyp2d6_substrate_carbonmangels':.10}

def med_trees(root,task,family,seed):
    vals=[]
    for part in ('fold0','fold1','fold2','valid'):
        p=root/task/part/family/str(seed)/'metadata.json'
        vals.append(json.loads(p.read_text())['trees'])
    return max(1,int(np.median(vals))),vals

def independent_reload_structure(d,v,k,task,family,iterations,seed,p):
    fit,q=e.rows(d,k,'final');folder=e.OUT/task/'final'/family/str(seed)
    meta=json.loads((folder/'metadata.json').read_text())
    model=(__import__('catboost').CatBoostClassifier if d['metrics'][k] in ('AUROC','AUPRC') else __import__('catboost').CatBoostRegressor)()
    model.load_model(str(folder/'model.cbm'))
    assert model.tree_count_==meta['trees']==iterations
    qx=e.matrix(d,v,q,family);metric=d['metrics'][k]
    z=model.predict_proba(qx)[:,1] if metric in ('AUROC','AUPRC') else model.predict(qx)
    if metric=='MAE':
        z=z*meta['target']['scale']+meta['target']['center']+e.prior(d,v,k,q,family)
    np.testing.assert_allclose(z,p,atol=1e-8,rtol=1e-8)
    return dict(task=task,family=family,seed=seed,trees=model.tree_count_,
                model_sha256=b.sha(folder/'model.cbm'),prediction_sha256=b.sha(folder/'predictions.npz'),
                max_error=float(np.max(np.abs(z-p))))

def independent_reload_conditional(d,k,task,iterations,seed,p):
    fit,q=c.rows(d,k,'final');folder=c.OUT/task/'final'/'aux'/str(seed)
    meta=json.loads((folder/'metadata.json').read_text())
    from catboost import CatBoostClassifier,CatBoostRegressor
    metric=d['metrics'][k];model=(CatBoostClassifier if metric in ('AUROC','AUPRC') else CatBoostRegressor)()
    model.load_model(str(folder/'model.cbm'));assert model.tree_count_==meta['trees']==iterations
    x,qx,audit=c.design(d,k,'final',fit,q,'aux');z=model.predict_proba(qx)[:,1] if metric in ('AUROC','AUPRC') else model.predict(qx)
    if metric=='MAE':z=z*meta['target']['scale']+meta['target']['center']
    np.testing.assert_allclose(z,p,atol=1e-8,rtol=1e-8)
    return dict(task=task,view='aux',seed=seed,trees=model.tree_count_,model_sha256=b.sha(folder/'model.cbm'),prediction_sha256=b.sha(folder/'predictions.npz'),max_error=float(np.max(np.abs(z-p))))

def main():
    d=b.build();v=e.features(d);base=ROOT/'admet_submissions/v72_v56/我从山北来带着兰花草-小分子药物预测技术创新赛-20261021.csv'
    template=ROOT/'admet_official/submission_template.csv';assert b.sha(base)=='9ac57cda636ea4cd7e8909e97ecde4ec656e51f024a73cd0fb6eaa29dd55f8b2'
    base_rows=[]
    with base.open(encoding='utf-8-sig',newline='') as f:base_rows=list(csv.reader(f))
    selected={};reload=[]
    for task,(family,w) in STRUCTURE.items():
        k=d['tasks'].index(task);iterations,tree_counts=med_trees(e.OUT,task,family,e.SEED);ps=[]
        for seed in SEEDS:
            ps.append(e.fit_predict(d,v,k,'final',family,iterations,seed))
            reload.append(independent_reload_structure(d,v,k,task,family,iterations,seed,ps[-1]))
        selected[task]=dict(kind='structure',family=family,weight=w,iterations=iterations,development_trees=tree_counts,prediction=np.mean(ps,axis=0))
    for task,w in CONDITIONAL.items():
        k=d['tasks'].index(task);iterations,tree_counts=med_trees(c.OUT,task,'aux',c.SEED);ps=[]
        for seed in SEEDS:
            ps.append(c.fit_predict(d,k,'final','aux',iterations,seed))
            reload.append(independent_reload_conditional(d,k,task,iterations,seed,ps[-1]))
        selected[task]=dict(kind='conditional',view='aux',weight=w,iterations=iterations,development_trees=tree_counts,prediction=np.mean(ps,axis=0))
    # Map selected model query rows to official task blocks and blend with v72.
    out_rows=[r[:] for r in base_rows];replay={}
    for task,rec in selected.items():
        k=d['tasks'].index(task);metric=d['metrics'][k];q=e.rows(d,k,'final')[1] if rec['kind']=='structure' else c.rows(d,k,'final')[1]
        pos=[i for i,r in enumerate(out_rows) if i and r[0]==task]
        assert len(pos)==len(q)==len(rec['prediction'])
        old=np.array([float(out_rows[i][4]) for i in pos]);new=b.blend(metric,old,rec['prediction'],rec['weight'])
        for i,x in zip(pos,new):out_rows[i][4]=repr(float(x))
        replay[task]=dict(kind=rec['kind'],family=rec.get('family',rec.get('view')),weight=rec['weight'],changed_rows=int(np.sum(new!=old)),baseline_sha256=hashlib.sha256(old.tobytes()).hexdigest(),prediction_sha256=hashlib.sha256(rec['prediction'].tobytes()).hexdigest(),result_sha256=hashlib.sha256(new.tobytes()).hexdigest())
    raw_template=template.read_bytes();raw_base=base.read_bytes();spans=field_spans(raw_template);chunks=[];cursor=0
    for i,row in enumerate(out_rows[1:],1):
        ref=base_rows[i];assert row[:4]+row[5:]==ref[:4]+ref[5:]
        lo,hi=spans[i][4];value=row[4].encode('ascii');assert not any(x in value for x in b',\r\n"')
        chunks.extend((raw_template[cursor:lo],value));cursor=hi
    chunks.append(raw_template[cursor:]);raw=b''.join(chunks)
    # Replacing predictions must leave every non-prediction byte equal to the template.
    actual=field_spans(raw);recovered=[];cursor=0
    for i in range(1,len(out_rows)):
        lo,hi=actual[i][4];a,z=spans[i][4];recovered.extend((raw[cursor:lo],raw_template[a:z]));cursor=hi
    recovered.append(raw[cursor:]);assert b''.join(recovered)==raw_template
    dest=ROOT/'admet_submissions/v73_v72';dest.mkdir(parents=True,exist_ok=True);file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261021.csv';file.write_bytes(raw)
    audit=verify(file,template);audit.update(status='PASS',baseline='v72',changed_tasks=replay,nonprediction_bytes_exact=True,automatic_platform_submission=False,official_score=None)
    b.save(file.with_suffix('.verification.json'),audit)
    release=dict(status='CANDIDATE_AWAITING_PLATFORM_EVALUATION',baseline='v72',baseline_official_score_user_reported=50.628,selected={k:{x:y for x,y in v.items() if x!='prediction'} for k,v in selected.items()},replay=replay,reload=reload,file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),template_sha256=b.sha(template),test_labels_used=False,external_data=False,conditional_tasks=list(CONDITIONAL),limitations='Selection reused official development labels and historical proxy controls. Conditional endpoint uses only official train+valid labels for other assays, with target values and masks absent; it is not pure structure-only evaluation. Official score is unknown until platform submission.')
    b.save(e.OUT/'release.json',release);b.save(e.OUT/'fusion_replay.json',replay)
    print(json.dumps(dict(status='EXPORTED',file=str(file),sha256=b.sha(file),changed_tasks=list(replay),reloads=len(reload)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
