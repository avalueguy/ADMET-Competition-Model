"""CPU-only postmortem. Does not train, change predictions, or submit anything."""
import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'admet_diagnostics/v21_postmortem'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(name, y, p):
    if name == 'MAE':
        return float(mean_absolute_error(y, p))
    if name == 'AUROC':
        return float(roc_auc_score(y, p))
    if name == 'AUPRC':
        return float(average_precision_score(y, p))
    return float(spearmanr(y, p).statistic)


def blend(a, b, weight, mode):
    if mode == 'rank':
        a = (rankdata(a) - .5) / len(a)
        b = (rankdata(b) - .5) / len(b)
    return (1-weight)*a + weight*b


def ranges(x):
    x = np.asarray(x)
    return dict(quantiles=list(map(float, np.quantile(x, [0, .25, .5, .75, 1]))),
                std=float(x.std()), unique=len(np.unique(x)))


def files():
    proof = read(ROOT/'admet_submissions/v21/verification.json')
    basename = Path(proof['submission']).name
    dest = ROOT/'admet_submissions/v21'/basename
    restored = False
    if not dest.exists():
        # Recover one exact regular file, never extract arbitrary archive paths.
        with tarfile.open(ROOT/'admet_v21_review.tar.gz', 'r:gz') as tar:
            member = tar.getmember('admet_submissions/v21/' + basename)
            assert member.isfile()
            data = tar.extractfile(member).read()
        assert hashlib.sha256(data).hexdigest() == proof['sha256']
        dest.write_bytes(data)
        restored = True
    baseline = next((ROOT/'admet_submissions/v20').glob('*.csv'))
    assert sha(baseline) == proof['baseline_sha256']
    assert sha(dest) == proof['sha256']
    frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in
              [baseline, dest, ROOT/'admet_official/submission_template.csv']]
    a, b, template = frames
    cols = ['task','sample_idx','drug_id','smiles','y_pred','metric']
    for frame in frames:
        assert list(frame.columns) == cols and len(frame) == 16379
        assert not frame.duplicated(['task','sample_idx']).any()
        assert frame.drop(columns='y_pred').equals(template.drop(columns='y_pred'))
    raw = dest.read_bytes()
    assert raw.startswith(b'\xef\xbb\xbf')
    assert raw.count(b'\r\n') == raw.count(b'\n') == 16380
    result = []
    for task, subset in b.groupby('task', sort=True):
        old = a.loc[a.task.eq(task), 'y_pred'].to_numpy(float)
        new = subset.y_pred.to_numpy(float)
        assert np.isfinite(new).all()
        assert np.array_equal(subset.sample_idx.astype(int), np.arange(len(new)))
        met = subset.metric.iloc[0]
        if met in ['AUROC','AUPRC']:
            assert np.all((new>=0)&(new<=1))
        changed = int(np.sum(old != new))
        r = dict(task=task, metric=met, rows=len(new), changed=changed)
        if changed:
            old_order = np.sign(old[:,None]-old[None,:])
            new_order = np.sign(new[:,None]-new[None,:])
            upper = np.triu(np.ones(old_order.shape, bool), 1)
            r.update(mean_abs_delta=float(np.abs(new-old).mean()), max_abs_delta=float(np.abs(new-old).max()),
                     spearman=float(spearmanr(old,new).statistic),
                     strict_pair_reversals=int(((old_order*new_order < 0)&upper).sum()),
                     pair_order_changes=int(((old_order != new_order)&upper).sum()),
                     old=ranges(old), new=ranges(new))
            s = read(ROOT/'admet_model_outputs_v21'/task/'selection.json')['selection']
            candidate = pd.read_csv(ROOT/'admet_model_outputs_v21'/task/'finetune_test.csv').Y_pred.to_numpy()
            expected = blend(old, candidate, s['weight'], s['mode'])
            np.testing.assert_allclose(new, expected, rtol=1e-12, atol=1e-12)
            r['fusion_replayed'] = True
        result.append(r)
    report = dict(baseline=str(baseline), candidate=str(dest), baseline_sha256=sha(baseline),
                  candidate_sha256=sha(dest), restored_from_archive_this_run=restored,
                  candidate_bytes=len(raw), metadata_exact=True, rows=len(b), tasks=result,
                  changed_rows=sum(r['changed'] for r in result),
                  changed_tasks=sum(r['changed']>0 for r in result), automatic_submission=False)
    save('files.json', report)
    print(json.dumps({'file_audit': {k:v for k,v in report.items() if k != 'tasks'},
                      'changes':[r for r in result if r['changed']]}, ensure_ascii=False), flush=True)


def data():
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
    template = pd.read_csv(ROOT/'admet_official/submission_template.csv', dtype=str, keep_default_na=False)
    results = []
    cache = {}
    def molecule(s):
        if s not in cache:
            m = Chem.MolFromSmiles(s)
            if m is None:
                cache[s] = None
            else:
                iso = Chem.MolToSmiles(m, isomericSmiles=True)
                non = Chem.MolToSmiles(m, isomericSmiles=False)
                scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
                cache[s] = (iso, non, scaf or 'ACYCLIC:'+iso, bool('@' in iso or '/' in iso or '\\' in iso), not bool(scaf))
        return cache[s]
    for task, section in template.groupby('task', sort=True):
        path = ROOT/'admet_competition_data/data_benchmark'/task
        frames = {s:pd.read_csv(path/(s+'.csv'), dtype={'Drug_ID':str,'Drug':str}, keep_default_na=False,
                              usecols=['Drug_ID','Drug']+(['Y'] if s != 'test' else []))
                  for s in ['train','valid','test']}
        info = {s:[molecule(str(x)) for x in f.Drug] for s,f in frames.items()}
        row = dict(task=task, metric=section.metric.iloc[0], splits={}, overlaps={})
        for split, f in frames.items():
            meta = info[split]
            good = [m for m in meta if m is not None]
            r = dict(rows=len(f), invalid_smiles=len(meta)-len(good),
                     duplicate_drug_ids=int(f.Drug_ID.duplicated().sum()),
                     duplicate_canonical_rows=len(good)-len({m[0] for m in good}),
                     unique_nonisomeric=len({m[1] for m in good}),
                     stereochemical_rows=sum(m[3] for m in good), acyclic_rows=sum(m[4] for m in good),
                     unique_scaffold_groups=len({m[2] for m in good}))
            if split != 'test':
                y=f.Y.to_numpy(float)
                r['nonfinite_labels']=int((~np.isfinite(y)).sum())
                r['label_quantiles']=list(map(float,np.quantile(y,[0,.25,.5,.75,1])))
                if row['metric'] in ['AUROC','AUPRC']:
                    r['positive_count']=int((y==1).sum())
                    r['positive_rate']=float(y.mean())
                    assert set(y)<= {0.,1.}
                dup={}
                for m,val in zip(meta,y):
                    if m: dup.setdefault(m[0],[]).append(float(val))
                r['canonical_conflicting_label_groups']=sum(len(set(v))>1 for v in dup.values())
                if row['metric']=='MAE':
                    r['observed_duplicate_only_mae_floor']=sum(sum(abs(z-np.median(v)) for z in v) for v in dup.values())/len(y)
                    r['floor_caveat']='Within-split oracle median per identical molecule, not an achievable validation model.'
                non_groups={}
                for m,val in zip(meta,y):
                    if m: non_groups.setdefault(m[1],[]).append((m[0],float(val)))
                r['nonisomeric_groups_with_multiple_stereoisomers']=sum(len({p[0] for p in v})>1 for v in non_groups.values())
                r['nonisomeric_alias_groups_with_different_labels']=sum(len({p[0] for p in v})>1 and len({p[1] for p in v})>1 for v in non_groups.values())
            row['splits'][split]=r
        for a,b in [('train','valid'),('train','test'),('valid','test')]:
            row['overlaps'][a+'_'+b] = {name:len({m[j] for m in info[a] if m}&{m[j] for m in info[b] if m})
                                                   for j,name in enumerate(['canonical','nonisomeric','scaffold'])}
        a=frames['test']; sec=section.reset_index(drop=True)
        assert a.Drug.tolist()==sec.smiles.tolist()
        assert a.Drug_ID.astype(str).tolist()==sec.drug_id.tolist()
        row['test_metadata_aligned']=True
        foldpath=ROOT/'admet_model_outputs_v5'/task/'folds.csv'
        if foldpath.exists():
            folds=pd.read_csv(foldpath,keep_default_na=False)
            assert folds.canonical.tolist()==[m[0] for m in info['train']]
            assert folds.scaffold.tolist()==[m[2] for m in info['train']]
            row['oof_fold_checks']=[]
            for f in sorted(folds.fold.unique()):
                left=folds.fold.eq(f); right=~left
                non=np.array([m[1] for m in info['train']])
                row['oof_fold_checks'].append(dict(fold=int(f),rows=int(left.sum()),
                    canonical_overlap=len(set(folds.canonical[left])&set(folds.canonical[right])),
                    scaffold_overlap=len(set(folds.scaffold[left])&set(folds.scaffold[right])),
                    nonisomeric_overlap=len(set(non[left])&set(non[right]))))
        results.append(row)
        print(json.dumps({'data_task_complete':task}),flush=True)
    summary={s:sum(r['splits'][s]['rows'] for r in results) for s in ['train','valid','test']}
    summary['unique_smiles_all_splits']=len(cache)
    summary['test_labels_read']=False
    save('data.json',dict(summary=summary,tasks=results))
    print(json.dumps(summary),flush=True)


def frontend():
    patterns=[r'.{0,150}(?:score|Score|排名|得分|分数|成绩).{0,200}',
              r'.{0,150}(?:setUpUrl\(e\)|uploadTime|aiaeScore).{0,250}']
    results=[]
    folder=ROOT/'admet_diagnostics/upload_web_audit'
    for p in sorted(folder.glob('*.js')):
        if p.name.startswith('chunk-vendors'): continue
        text=p.read_text(encoding='utf-8')
        hits=[]
        for pattern in patterns:
            hits.extend(m.group() for m in re.finditer(pattern,text))
        if hits:
            results.append(dict(file=p.name,sha256=sha(p),snippets=list(dict.fromkeys(hits))))
    save('frontend.json',results)
    for r in results:
        print(json.dumps(r,ensure_ascii=False),flush=True)


def remote():
    # These entry points only reconstruct stored predictions; no fit/predict calls.
    from admet_v21_anchor import anchor_oof, anchor_valid
    template=pd.read_csv(ROOT/'admet_official/submission_template.csv',dtype=str)
    results=[]
    for task,section in template.groupby('task',sort=True):
        try:
            y,fold,b=anchor_oof(task)
            vy,vb=anchor_valid(task)
        except FileNotFoundError as exc:
            results.append(dict(task=task,status='baseline_unavailable',error=str(exc)))
            continue
        met=section.metric.iloc[0]
        row=dict(task=task,metric=met,baseline_valid=metric(met,vy,vb),baseline_oof=[])
        d=ROOT/'admet_model_outputs_v21'/task
        if (d/'selection.json').exists():
            s=read(d/'selection.json')['selection']
            c=pd.read_csv(d/'finetune_oof.csv')
            cv=pd.read_csv(d/'finetune_valid.csv')
            np.testing.assert_array_equal(c.Y_true,y)
            np.testing.assert_array_equal(c.fold,fold)
            np.testing.assert_array_equal(cv.Y_true,vy)
            row.update(selection=s,pure_candidate_valid=metric(met,vy,cv.Y_pred),
                       valid_baseline_range=ranges(vb),valid_candidate_range=ranges(cv.Y_pred),
                       candidate_valid_selected=metric(met,vy,blend(vb,cv.Y_pred,s['weight'],s['mode'])),
                       folds=[])
            if (d/'finetune_test.csv').exists():
                bc=pd.read_csv(next((ROOT/'admet_submissions/v20').glob('*.csv')))
                tb=bc.loc[bc.task.eq(task),'y_pred'].to_numpy(float)
                tc=pd.read_csv(d/'finetune_test.csv').Y_pred.to_numpy(float)
                row['test_baseline_range']=ranges(tb)
                row['test_candidate_range']=ranges(tc)
            for stage in ['fold0','fold1','fold2','valid','final','repeat']:
                p=d/stage/'metadata.json'
                if p.exists():
                    meta=read(p)
                    records=[]
                    for f,h in meta['files'].items():
                        assert sha(p.parent/f)==h,(task,stage,f)
                        records.append(f)
                    row.setdefault('stages',[]).append(dict(stage=stage,fit_rows=meta['fit_rows'],query_rows=meta['query_rows'],
                        epochs=meta['epochs'],model_files_hash_verified=records,stored_replay_passed=meta['model_replay_passed']))
        for f in sorted(np.unique(fold)):
            mask=fold==f
            row['baseline_oof'].append(dict(fold=int(f),value=metric(met,y[mask],b[mask])))
            if 'selection' in row:
                cp=c.Y_pred.to_numpy()[mask]
                row['folds'].append(dict(fold=int(f),pure_candidate=metric(met,y[mask],cp),
                    selected=metric(met,y[mask],blend(b[mask],cp,s['weight'],s['mode'])),
                    baseline_range=ranges(b[mask]),candidate_range=ranges(cp)))
        results.append(row)
    save('remote_metrics.json',results)
    print(json.dumps({'tasks':len(results),'hash_verified_stages':sum(len(r.get('stages',[])) for r in results)}),flush=True)


def initialization():
    # CPU initialization only; no forward/backward, optimizer, CUDA or training.
    import torch
    import admet_v21_finetune as ft
    torch.set_num_threads(4)
    assert ft.BASE is None
    torch.manual_seed(21271)
    cold=ft.TailModel(367)
    h1=cold.head[0].weight.detach().clone()
    torch.manual_seed(21271)
    warm=ft.TailModel(367)
    h2=warm.head[0].weight.detach().clone()
    torch.manual_seed(21271)
    warm2=ft.TailModel(367)
    h3=warm2.head[0].weight.detach().clone()
    result=dict(seed=21271,device='cpu',training_performed=False,
                cold_equals_warm=bool(torch.equal(h1,h2)),warm_equals_warm=bool(torch.equal(h2,h3)),
                cold_warm_max_abs_difference=float((h1-h2).abs().max()),
                cold_head_sha256=hashlib.sha256(h1.numpy().tobytes()).hexdigest(),
                warm_head_sha256=hashlib.sha256(h2.numpy().tobytes()).hexdigest(),
                encoder_dropouts={name:module.p for name,module in cold.layers.named_modules() if isinstance(module,torch.nn.Dropout)},
                note='Cold pretrained() overrides the caller seed. Impact depends on entry point and initialization order; not a demonstrated cause of the platform result.')
    assert not result['cold_equals_warm'] and result['warm_equals_warm']
    save('initialization.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['files','data','frontend','remote','initialization'])
    args=ap.parse_args()
    globals()[args.stage]()
