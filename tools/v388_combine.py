"""Compose predeclared nonoverlapping whole-task updates, with no fitting."""
from pathlib import Path
from datetime import datetime
import csv, hashlib, io, json
import numpy as np
from scipy.stats import spearmanr
from repack_upload_template import field_spans
from verify_submission import verify

ROOT=Path(__file__).resolve().parents[1]
TASKS=('caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb','clearance_hepatocyte_az','clearance_microsome_az','half_life_obach','vdss_lombardo')
PARTS=('fold0','fold1','fold2','valid')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(raw):return list(csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline='')))


def main():
    protocol=read(ROOT/'docs/v388_union_protocol.json')
    launch=read(ROOT/'experiments/regression9_v387/outputs/protocol.json')
    status=read(ROOT/'experiments/regression9_v387/run_status.json')
    assert status['stage']=='EVIDENCE_READY'
    audit=read(ROOT/'docs/v388_remote_preregistration_audit.json')
    assert audit['status']=='PASS'and audit['protocol_sha256']==sha(ROOT/'docs/v388_union_protocol.json')
    assert datetime.fromisoformat(protocol['registered_at_utc']).timestamp()<audit['launch_mtime_unix']
    assert launch['signature']==status['signature']
    best=read(ROOT/protocol['baseline']);basepath=ROOT/best['csv']
    assert sha(basepath)==best['csv_sha256']==protocol['baseline_csv_sha256']
    source={v:read(ROOT/f'docs/{v}_delivery.json')for v in protocol['source_priority']}
    selected={}
    for v,d in source.items():
        assert d['baseline']['csv_sha256']==best['csv_sha256']
        for row in d['task_evidence']:
            if row['accepted'] and row['task'] not in selected:
                selected[row['task']]=dict(source=v,evidence=row)
    added=[t for t,r in selected.items()if r['source']=='v387']
    both=bool(source['v386']['csv'] and added)
    means={p:sum(next(x['gain']for x in r['evidence']['parts']if x['part']==p)for r in selected.values())/9 for p in PARTS}
    five={p:sum(next(x['gain']for x in r['evidence']['parts']if x['part']==p)for t,r in selected.items()if t in TASKS[:5])/5 for p in PARTS}
    result=dict(version='v388',status='NO_UNION_NEEDED',protocol='docs/v388_union_protocol.json',protocol_sha256=sha(ROOT/'docs/v388_union_protocol.json'),
        csv=None,official_best=51.308,new_official_score=None,baseline=best,selected_tasks=selected,
        nine_task_mean_relative_MAE_gain_by_part=means,five_scored_task_mean_relative_MAE_gain_by_part=five,
        goal_mean_5percent_achieved_local=all(x>=.05 for x in means.values()),five_scored_goal_achieved_local=all(x>=.05 for x in five.values()),
        no_fitting_performed=True,automatic_platform_submission=False,overlap_rule=protocol['overlap_rule'])
    if both:
        raw=basepath.read_bytes();tab=table(raw);spans=field_spans(raw);replace={};changed={}
        for t,r in selected.items():
            v=r['source'];d=source[v];path=ROOT/d['csv'];assert sha(path)==d['csv_sha256'];other=table(path.read_bytes())
            assert len(other)==len(tab);exp=ROOT/f'experiments/regression9_{v}'
            manifest=read(exp/'data_manifest.json');assert manifest['baseline_csv_sha256']==best['csv_sha256']
            data=np.load(exp/'official_rows.npz');e=r['evidence'];conf=read(exp/'outputs/confirmation.json')
            c=next(x for x in conf['records']if x['task']==t);assert c['experimental_eligible']
            for p in PARTS:
                a=np.load(exp/'controls'/t/(p+'.npz'));z=np.load(exp/'outputs/tasks'/t/p/e['best']['name']/'predictions.npz')
                for k in('fit','query'):np.testing.assert_array_equal(a[k],z[k])
                y=data['y'][a['query']];pred=(1-e['weight'])*a['p']+e['weight']*z['p']
                gain=float((np.abs(y-a['p']).mean()-np.abs(y-pred).mean())/np.abs(y-a['p']).mean())
                reported=next(x for x in e['parts']if x['part']==p);assert abs(gain-reported['gain'])<1e-12 and gain>0
                if t in TASKS[5:]:
                    rank=float(spearmanr(y,pred).statistic-spearmanr(y,a['score_p']).statistic)
                    assert abs(rank-reported['rank_gain'])<1e-12 and rank>0
            a=np.load(exp/'controls'/t/'final.npz');z=np.load(exp/'outputs/tasks'/t/'final'/e['best']['name']/'predictions.npz')
            loc=[i for i,row in enumerate(tab)if i and row[0]==t]
            pred=(1-e['weight'])*a['p']+e['weight']*z['p']
            assert np.isnan(data['y'][a['query']]).all() and np.isfinite(pred).all()
            for i,prediction in zip(loc,pred):
                assert other[i][:4]+other[i][5:]==tab[i][:4]+tab[i][5:]
                assert float(other[i][4])==float(prediction)
                replace[i]=other[i][4].encode('ascii')
        chunks=[];cursor=0
        for i,b in sorted(replace.items()):
            lo,hi=spans[i][4];chunks.extend([raw[cursor:lo],b]);cursor=hi
        chunks.append(raw[cursor:]);payload=b''.join(chunks)
        folder=ROOT/'admet_submissions/v388_v385_task_experts_experimental';folder.mkdir(exist_ok=True)
        path=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261106.csv'
        if path.exists():assert path.read_bytes()==payload
        else:path.write_bytes(payload)
        check=verify(path,ROOT/'admet_official/submission_template.csv')
        for i,(old,new)in enumerate(zip(tab,table(payload))):
            assert old[:4]+old[5:]==new[:4]+new[5:]
            if i not in replace:assert old==new
            if old!=new:changed[old[0]]=changed.get(old[0],0)+1
        result.update(status='VERIFIED_EXPERIMENTAL_DELIVERY',csv=path.relative_to(ROOT).as_posix(),csv_sha256=sha(path),changed_rows=changed,
            unchanged_tasks=22-len(selected),strict_stability_confirmed=all(r['evidence']['strict_accepted']for r in selected.values()),format=check)
        save(path.with_suffix('.verification.json'),result)
    else:
        result['recommended_source']=next((v for v in protocol['source_priority']if source[v]['csv']),None)
    save(ROOT/'docs/v388_delivery.json',result)
    print(json.dumps({k:v for k,v in result.items()if k not in('selected_tasks','baseline','format')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
