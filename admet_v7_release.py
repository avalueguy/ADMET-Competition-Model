"""Frozen-baseline release export, independent inference and compact evidence."""
import argparse
import json
import shutil
import tarfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from admet_v6_runner import remote_only,save,sha,emit,metrics,kind,deltas,secondary_ok
from admet_v7_runner import OUT,protocol
from admet_v7_models import load_task,fit,predict
from admet_submission import build

ROOT=Path('admet_release_v7')
UPLOAD=Path('admet_submissions/v7/我从山北来带着兰花草-小分子药物预测技术创新赛-20260925.csv')


def publish():
    p=protocol();ROOT.mkdir(exist_ok=True);routes=[]
    for prior in sorted(x for x in Path('admet_release_v6').iterdir() if x.is_dir()):
        task=prior.name;dest=ROOT/task;dest.mkdir(exist_ok=True)
        decision=json.loads((OUT/task/'decision.json').read_text());accepted=decision['accepted']
        if accepted:
            d=load_task(task);n=len(d['frames']['train'])+len(d['frames']['valid']);y=np.r_[d['frames']['train'].Y,d['frames']['valid'].Y]
            s=decision['selection'];modelpath=dest/'model.joblib'
            previous=json.loads((dest/'selection.json').read_text()) if (dest/'selection.json').exists() else {}
            if modelpath.exists() and previous.get('model_sha256')==sha(modelpath) and previous.get('selection')==s:model=joblib.load(modelpath)
            else:
                model=fit(s['family'],d,np.arange(n),y,s['metric'],p);joblib.dump(model,modelpath,compress=3)
            pred=predict(model,d,np.arange(n,len(d['smiles'])));old=pd.read_csv(prior/f'submission_{task}.csv',dtype={'Drug_ID':str},keep_default_na=False)
            q=(1-s['alpha'])*old.Y.to_numpy(float)+s['alpha']*pred
            assert np.isfinite(q).all()
            if s['kind']=='classification':assert ((q>=0)&(q<=1)).all()
            pd.DataFrame(dict(Drug_ID=old.Drug_ID,Y=q)).to_csv(dest/f'submission_{task}.csv',index=False)
            pd.DataFrame(dict(Drug_ID=old.Drug_ID,Y=pred)).to_csv(dest/'candidate_test.csv',index=False)
            v=pd.read_csv(OUT/task/'validation_comparison.csv');pd.DataFrame(dict(Y_true=v.Y_true,Y_pred=v.blend)).to_csv(dest/'validation_predictions.csv',index=False)
            decision.update(model_sha256=sha(modelpath),feature_key=d['v7_feature_key'])
        else:
            for file in [f'submission_{task}.csv','validation_predictions.csv']:shutil.copy2(prior/file,dest/file)
        decision.update(base_sha256=sha(prior/f'submission_{task}.csv'),prediction_sha256=sha(dest/f'submission_{task}.csv'))
        save(dest/'selection.json',decision);routes.append(decision);emit(stage='v7_publish',task=task,accepted=accepted)
    save(ROOT/'routing.json',routes)
    accepted=[r['task'] for r in routes if r['accepted']]
    if not accepted:
        save(ROOT/'verification.json',dict(status='NO_PROMOTION',baseline_score=49.247501,accepted=[]));return
    build(ROOT,Path('admet_submissions/admet-v7-20260925.csv'))
    df=pd.read_csv('admet_submissions/admet-v7-20260925.csv',dtype={'drug_id':str},keep_default_na=False)
    UPLOAD.parent.mkdir(parents=True,exist_ok=True);df.to_csv(UPLOAD,index=False,encoding='utf-8-sig',lineterminator='\r\n')
    save(ROOT/'verification.json',dict(status='EXPORTED_AWAITING_REPLAY',accepted=accepted,submission=str(UPLOAD),sha256=sha(UPLOAD),
        baseline_score=49.247501,platform_score=None,rows=len(df),protocol_sha256=sha('admet_v7_protocol.json')))
    save(ROOT/'source_manifest.json',[dict(path=str(f),sha256=sha(f)) for f in sorted(Path('.').glob('admet_v7*')) if f.is_file()])


def verify():
    v=json.loads((ROOT/'verification.json').read_text())
    if v['status']=='NO_PROMOTION':save(ROOT/'independent_verification.json',v);return
    upload=Path(v['submission']);blob=upload.read_bytes();assert sha(upload)==v['sha256']
    assert blob.startswith(b'\xef\xbb\xbf') and blob.count(b'\r\n')==16380
    assert blob.count(b'task,sample_idx,drug_id,smiles,y_pred,metric')==1
    new=pd.read_csv(upload,dtype={'drug_id':str},keep_default_na=False)
    template=pd.read_csv('admet_official/submission_template.csv',dtype={'drug_id':str},keep_default_na=False)
    assert new.drop(columns='y_pred').equals(template.drop(columns='y_pred'))
    assert len(new)==16379 and new.task.nunique()==22
    checks=[]
    for folder in sorted(f for f in ROOT.iterdir() if f.is_dir()):
        task=folder.name;r=json.loads((folder/'selection.json').read_text());oldfile=Path('admet_release_v6')/task/f'submission_{task}.csv';file=folder/f'submission_{task}.csv'
        assert sha(oldfile)==r['base_sha256'];assert sha(file)==r['prediction_sha256'];result=pd.read_csv(file)
        if r['accepted']:
            s=r['selection'];metric=s['metric'];kk=s['kind'];tolerance=.01 if metric=='MAE' else .005
            oof=pd.read_csv(OUT/task/'baseline_oof.csv');candidate=pd.read_csv(OUT/task/(s['family']+'_oof.csv'))
            np.testing.assert_allclose(candidate.Y_true,oof.Y_true);np.testing.assert_array_equal(candidate.fold,oof.fold)
            gains=[]
            for fold in range(3):
                mask=oof.fold.eq(fold);y=oof.loc[mask,'Y_true'].to_numpy();b=oof.loc[mask,'Y_pred'].to_numpy();p=candidate.loc[mask,'Y_pred'].to_numpy()
                before=metrics(kk,y,b);after=metrics(kk,y,(1-s['alpha'])*b+s['alpha']*p)
                gains.append(deltas(metric,before,after));assert gains[-1]>=-tolerance and secondary_ok(metric,before,after)
            assert np.mean(gains)>0
            for split,file in [('valid','validation_comparison.csv'),('repeat','repeat_comparison.csv')]:
                frame=pd.read_csv(OUT/task/file);before=metrics(kk,frame.Y_true,frame.baseline);after=metrics(kk,frame.Y_true,frame.blend)
                np.testing.assert_allclose(frame.blend,(1-s['alpha'])*frame.baseline+s['alpha']*frame.candidate,atol=1e-12)
                assert deltas(metric,before,after)>=(0 if split=='valid' else -tolerance) and secondary_ok(metric,before,after)
            assert all(x['passed'] for x in r['domains'])
            assert sha(folder/'model.joblib')==r['model_sha256'];d=load_task(task);n=len(d['frames']['train'])+len(d['frames']['valid'])
            pred=predict(joblib.load(folder/'model.joblib'),d,np.arange(n,len(d['smiles'])))
            np.testing.assert_allclose(pred,pd.read_csv(folder/'candidate_test.csv').Y,rtol=3e-6,atol=3e-6)
            a=r['selection']['alpha'];np.testing.assert_allclose(result.Y,(1-a)*pd.read_csv(oldfile).Y+a*pred,rtol=3e-6,atol=3e-6)
        else:assert file.read_bytes()==oldfile.read_bytes()
        piece=new.loc[new.task==task];np.testing.assert_allclose(piece.y_pred,result.Y,rtol=1e-12,atol=1e-12)
        assert np.isfinite(piece.y_pred).all()
        if piece.metric.iloc[0] in ['AUROC','AUPRC']:assert piece.y_pred.between(0,1).all()
        checks.append(dict(task=task,accepted=r['accepted'],passed=True))
        emit(stage='v7_replay',task=task,passed=True)
    v.update(status='VERIFIED',independent_replay=True,task_checks=checks)
    save(ROOT/'independent_verification.json',v)


def report():
    from admet_v7_uncertainty import main as uncertainty
    uncertainty()
    routes=json.loads((ROOT/'routing.json').read_text());rows=[];lines=[
        '# v7 远程迭代结果','',
        'v6 平台成绩由用户报告为 **49.247501**，较 v5 增加 0.266221。目标 51.584929，差距 2.337428。v7 平台分数尚未知。','',
        '本轮新增原子对、拓扑扭转、MACCS 与理化描述符混合核，互补指纹 CatBoost，以及训练数据 SMILES 增强的 ChemBERTa 监督微调。实际入选方法见下表。','',
        '候选池及权重事先固定，先用三折骨架 OOF 锁定方案，再检查官方 valid、额外骨架切分及低相似度分组。未通过的任务逐字节保留 v6。','',
        '|任务|计分指标|v6 valid|v7 最终 valid|采用方法|新模型权重|','|---|---|---:|---:|---|---:|']
    from admet_v6_runner import metrics,kind,get_metric
    for r in routes:
        task=r['task'];metric=get_metric(task);old=pd.read_csv(Path('admet_release_v6')/task/'validation_predictions.csv');new=pd.read_csv(ROOT/task/'validation_predictions.csv')
        a=metrics(kind(metric),old.Y_true,old.Y_pred);b=metrics(kind(metric),new.Y_true,new.Y_pred);key=metric.lower()
        family=r['selection']['family'] if r['accepted'] else '保留 v6';alpha=r['selection']['alpha'] if r['accepted'] else 0
        lines.append(f"|{task}|{metric}|{a[key]:.6f}|{b[key]:.6f}|{family}|{alpha}|")
        rows.append(dict(task=task,metric=metric,accepted=r['accepted'],family=family,alpha=alpha,reason=r['reason'],**{'before_'+k:v for k,v in a.items()},**{'after_'+k:v for k,v in b.items()}))
    lines+=['','## 采用任务的辅助指标与额外骨架检查','',
        '|任务|辅助指标 v6 → v7|三折主指标最差变化|额外切分主指标 v6 → v7|',
        '|---|---|---:|---|']
    for r in routes:
        if not r['accepted']:continue
        metric=r['selection']['metric'];primary=metric.lower()
        secondary='spearman' if metric=='MAE' else ('mae' if metric=='Spearman' else ('auroc' if metric=='AUPRC' else 'auprc'))
        lines.append(f"|{r['task']}|{secondary}: {r['before'][secondary]:.6f} → {r['after'][secondary]:.6f}|{r['selection']['worst_gain']:.6f}|{r['repeat']['before'][primary]:.6f} → {r['repeat']['after'][primary]:.6f}|")
    lines+=['','MAE 最差变化为相对改善，排名指标为绝对改善；负数表示容差内退步。条件骨架 bootstrap 区间详见 `v7_uncertainty.csv`，仅描述已选预测的波动，未校正选型偏差。',
        '','## 风险与可复现范围','',
        '- 三折 OOF、官方 valid 和额外切分均有历史复用，本轮比较存在选择偏差；不能称独立测试，也不能保证隐藏集不退步。',
        '- 主指标折内最大容差为 MAE 1% / 排名指标 0.005；辅助指标容差为 MAE 3% / 排名指标 0.015。官方 valid 主指标要求不下降。',
        '- 低相似度组低于 20 条或缺少两类时，证据不足，报告明确标记；其他低相似度组限制主指标下降为 MAE 3% / 排名指标 0.01。',
        '- 官网未公开精确聚合公式和分项分数，不推算 v7 平台分数。',
        '- 全部训练在远程服务器；未使用盲测标签、未引入外部 ADMET 标签、未自动上传平台。',
        '- 本次继续依赖已冻结的历史 v6 预测，新候选保存检查点并在独立进程重放。',
        '- 随模型池扩大，过拟合风险增加；本轮不在 valid 失败后尝试第二名候选。','',
        '远程目录：`/root/autodl-tmp/admet_20260925`。本机只保存代码、轻量报告和提交文件。']
    Path('admet_diagnostics/v7_result_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    pd.DataFrame(rows).to_csv('admet_diagnostics/v7_comparison.csv',index=False)
    keep=[]
    for root in [ROOT,OUT]:
        keep += [p for p in root.rglob('*') if p.is_file() and p.suffix in ['.json','.csv'] and 'features' not in p.parts]
    keep += list(Path('admet_submissions/v7').glob('*')) if Path('admet_submissions/v7').exists() else []
    keep += list(Path('admet_diagnostics').glob('v7_*'))+list(Path('logs').glob('v7*.log'))
    with tarfile.open('admet_v7_review.tar.gz','w:gz') as tf:
        for p in keep:tf.add(p,arcname=str(p),recursive=False)
    emit(stage='v7_packaged',archive_bytes=Path('admet_v7_review.tar.gz').stat().st_size,accepted=[r['task'] for r in routes if r['accepted']])


if __name__=='__main__':
    remote_only();ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['publish','verify','report']);args=ap.parse_args()
    {'publish':publish,'verify':verify,'report':report}[args.stage]()
