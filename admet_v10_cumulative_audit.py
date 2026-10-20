"""Read-only score-anchor audit: evaluate the full v7-to-v10 recipe, not adjacent deltas.

No fitting, new selection, or prediction-file writes. Uses only saved predictions
and label-free similarity on historically reused validation structures.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from admet_v6_runner import remote_only, metrics, kind, deltas, secondary_ok, get_metric, save
from admet_v6_domain_guard import nearest
from admet_v8_models import load_task

ANCHOR_HASH='88e81b9d16e48cb941f18c5fd6c4a49e067c73659de808e896e51dea8de5acf0'
FINAL_HASH='d9d7a9e725d2083e799229e1c988ee396d598453d58c331c2143d77219ab9a4d'
NAME='我从山北来带着兰花草-小分子药物预测技术创新赛-20260925.csv'
SOURCES={}


def read_json(path):
    path=Path(path); content=path.read_bytes(); SOURCES[str(path)]=hashlib.sha256(content).hexdigest()
    return json.loads(content)


def read_csv(path):
    path=Path(path); SOURCES[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def compare(task, split, y, old, new, metric, distant=False):
    covered=len(y)>=20 and (kind(metric)!='classification' or len(np.unique(y))==2)
    result=dict(task=task,split=split,metric=metric,distant=distant,n=len(y),coverage_sufficient=covered,
                primary_pass=None,secondary_pass=None)
    if not covered:
        result['reason']='insufficient subgroup coverage; no safety conclusion';return result
    b=metrics(kind(metric),y,old);a=metrics(kind(metric),y,new)
    gain=deltas(metric,b,a)
    tol=(.03 if metric=='MAE' else .01) if distant else (.01 if metric=='MAE' else .005)
    if split=='valid' and not distant: tol=0
    secondary='spearman' if metric=='MAE' else ('mae' if metric=='Spearman' else ('auroc' if metric=='AUPRC' else 'auprc'))
    secondary_gain=(b[secondary]-a[secondary])/max(b[secondary],1e-8) if secondary=='mae' else a[secondary]-b[secondary]
    result.update(primary_gain=gain,primary_tolerance=tol,primary_pass=bool(gain>=-tol),
        secondary_metric=secondary,secondary_gain=secondary_gain,secondary_pass=bool(secondary_ok(metric,b,a)),
        **{'before_'+k:v for k,v in b.items()},**{'after_'+k:v for k,v in a.items()})
    return result


def evaluate_split(task,split,d,it,iv,y,old,new,metric):
    result=[compare(task,split,y,old,new,metric)]
    sim=nearest(d['raw'],it,iv);mask=sim<.3
    result.append(compare(task,split,y[mask],old[mask],new[mask],metric,True))
    return result


def main():
    remote_only()
    original=Path('admet_submissions/v7')/NAME; final=Path('admet_submissions/v10')/NAME
    assert hashlib.sha256(original.read_bytes()).hexdigest()==ANCHOR_HASH
    assert hashlib.sha256(final.read_bytes()).hexdigest()==FINAL_HASH
    old=pd.read_csv(original,dtype=str,keep_default_na=False);new=pd.read_csv(final,dtype=str,keep_default_na=False)
    assert new.drop(columns='y_pred').equals(old.drop(columns='y_pred'))
    changed=set(new.loc[new.y_pred!=old.y_pred,'task']); rows=[]; chains=[]; unchanged=[]
    for task in sorted(old.task.unique()):
        routes={v:read_json(f'admet_release_v{v}/{task}/selection.json') for v in [8,9,10]}
        adopted=[v for v,r in routes.items() if r['accepted']]
        assert bool(adopted)==(task in changed)
        v7=read_csv(f'admet_release_v7/{task}/validation_predictions.csv')
        vf=read_csv(f'admet_release_v10/{task}/validation_predictions.csv')
        np.testing.assert_allclose(v7.Y_true,vf.Y_true,rtol=1e-12,atol=1e-12)
        if not adopted:
            mask=old.task.eq(task); assert new.loc[mask,'y_pred'].tolist()==old.loc[mask,'y_pred'].tolist()
            np.testing.assert_array_equal(v7.Y_pred,vf.Y_pred);unchanged.append(task);continue
        d=load_task(task);metric=get_metric(task);n=len(d['frames']['train']);nv=len(d['frames']['valid'])
        oof=read_csv(f'admet_model_outputs_v8/{task}/baseline_oof.csv')
        np.testing.assert_allclose(oof.Y_true,d['frames']['train'].Y)
        anchor=oof.Y_pred.to_numpy().copy();cumulative=anchor.copy();val=v7.Y_pred.to_numpy().copy()
        it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=692509).split(np.arange(n),groups=d['groups'][:n]))
        repeat_anchor=None;repeat_current=None
        for v in [8,9,10]:
            r=routes[v]
            if r['accepted']:
                s=r['selection'];a=s['alpha'];p=read_csv(f'admet_model_outputs_v{v}/{task}/{s["family"]}_oof.csv')
                np.testing.assert_array_equal(p.row,oof.row);np.testing.assert_array_equal(p.fold,oof.fold);np.testing.assert_allclose(p.Y_true,oof.Y_true)
                cumulative=(1-a)*cumulative+a*p.Y_pred.to_numpy()
                hold=read_csv(f'admet_model_outputs_v{v}/{task}/validation_comparison.csv')
                np.testing.assert_allclose(hold.baseline,val,rtol=1e-12,atol=1e-12)
                # Neural candidates were multiplied in float32 before their saved blend.
                # Verify numerical reconstruction, then preserve exact recorded predictions.
                expected=(1-a)*val+a*hold.candidate.to_numpy()
                np.testing.assert_allclose(expected,hold.blend,rtol=1e-6,atol=1e-7)
                val=hold.blend.to_numpy().copy()
                rep=read_csv(f'admet_model_outputs_v{v}/{task}/repeat_comparison.csv')
                np.testing.assert_array_equal(rep.row,iv);np.testing.assert_allclose(rep.Y_true,d['frames']['train'].Y.to_numpy()[iv])
                if repeat_anchor is None:
                    repeat_anchor=rep.baseline.to_numpy().copy();repeat_current=repeat_anchor.copy()
                np.testing.assert_allclose(rep.baseline,repeat_current,rtol=1e-12,atol=1e-12)
                expected=(1-a)*repeat_current+a*rep.candidate.to_numpy()
                np.testing.assert_allclose(expected,rep.blend,rtol=1e-6,atol=1e-7)
                repeat_current=rep.blend.to_numpy().copy()
            release=read_csv(f'admet_release_v{v}/{task}/validation_predictions.csv')
            np.testing.assert_allclose(release.Y_pred,val,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(vf.Y_pred,val,rtol=1e-12,atol=1e-12)
        for f in range(3):
            tr=np.flatnonzero(oof.fold!=f);va=np.flatnonzero(oof.fold==f)
            rows.extend(evaluate_split(task,f'fold_{f}',d,tr,va,oof.Y_true.to_numpy()[va],anchor[va],cumulative[va],metric))
        rows.extend(evaluate_split(task,'valid',d,np.arange(n),np.arange(n,n+nv),v7.Y_true.to_numpy(),v7.Y_pred.to_numpy(),val,metric))
        rows.extend(evaluate_split(task,'repeat',d,it,iv,d['frames']['train'].Y.to_numpy()[iv],repeat_anchor,repeat_current,metric))
        chains.append(dict(task=task,adopted_versions=adopted))
        print(json.dumps(dict(stage='anchor_audit',task=task,adopted=adopted)),flush=True)
    violations=[x for x in rows if x['primary_pass'] is False or x['secondary_pass'] is False]
    insufficient=[x for x in rows if not x['coverage_sufficient']]
    result=dict(status='AUDITED',anchor_version=7,anchor_score=49.343821,target_score=51.584929,
        final_version=10,final_score=None,anchor_sha256=ANCHOR_HASH,final_sha256=FINAL_HASH,
        changed_tasks=sorted(changed),unchanged_tasks=unchanged,chains=chains,
        comparisons=len(rows),violations=violations,insufficient_coverage=insufficient,
        repeated_adoption_tasks=[x['task'] for x in chains if len(x['adopted_versions'])>1],
        limitation='Post-selection diagnostic on reused data; no new fitting, selection, or hidden-score inference.',
        sources=SOURCES)
    assert hashlib.sha256(final.read_bytes()).hexdigest()==FINAL_HASH
    save('admet_diagnostics/v10_anchor_audit.json',result)
    pd.DataFrame(rows).to_csv('admet_diagnostics/v10_anchor_audit.csv',index=False)
    lines=['# v10 相对平台已评分 v7 的累计回退审计','',
        '本检查将 v8、v9、v10 已采用的预测逐步重建，直接比较 v7 和最终 v10。没有重新训练、选择候选或修改预测文件。', '',
        f'累计改变 {len(changed)} 项任务，{len(unchanged)} 项预测逐字保留；执行 {len(rows)} 项整体/低相似度组比较。重复采用的任务：'+', '.join(result['repeated_adoption_tasks'])+'。', '',
        f'按整体主/辅助指标及低相似度组增强标准检查，发现 {len(violations)} 项超出容差；{len(insufficient)} 项覆盖不足，不能证明安全。', '',
        '|任务|切分|低相似度组|主指标通过|辅助指标通过|', '|---|---|---|---|---|']
    for x in violations:lines.append(f"|{x['task']}|{x['split']}|{x['distant']}|{x['primary_pass']}|{x['secondary_pass']}|")
    lines+=['','详细逐指标数字见 v10_anchor_audit.csv；完整来源哈希见 v10_anchor_audit.json。低相似度组的辅助指标门槛是 v10 新增检查，不能把此前发布 v8/v9 时未使用的标准描述为此前已经通过。', '',
        '验证划分反复使用，仍有选择偏差；通过检查不代表隐藏集无回退，也不能换算平台分数。最后平台实测仍是 49.343821，目标仍是 51.584929。','']
    Path('admet_diagnostics/v10_anchor_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
