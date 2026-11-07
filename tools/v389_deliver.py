"""Receive and independently check frozen fusion arithmetic, with no local fits."""
from pathlib import Path
import csv,hashlib,io,json,tarfile
import numpy as np
from scipy.stats import spearmanr
from repack_upload_template import field_spans
from verify_submission import verify
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/regression9_v389'
TASKS=('caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb','clearance_hepatocyte_az','clearance_microsome_az','half_life_obach','vdss_lombardo')
PARTS=('fold0','fold1','fold2','valid')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def table(b):return list(csv.reader(io.StringIO(b.decode('utf-8-sig'),newline='')))
def main():
    receipt=read(EXP/'result_archive.json');archive=ROOT/'releases/v389_remote_results.tar.gz';assert receipt['status']=='PASS'and sha(archive)==receipt['sha256']
    with tarfile.open(archive)as tf:
        seen=set()
        for m in tf.getmembers():
            dest=(EXP/m.name).resolve();assert m.isfile()and dest.is_relative_to(EXP.resolve())and m.name not in seen;seen.add(m.name)
            payload=tf.extractfile(m).read()
            if dest.exists():assert dest.read_bytes()==payload,m.name
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
    for r in read(EXP/'result_manifest.json')['files']:assert sha(EXP/r['path'])==r['sha256']
    plan=read(EXP/'plan.json');assert sha(ROOT/'tools/v389_combination.py')==sha(EXP/'v389_combination.py')==plan['source_sha256']
    verification=read(EXP/'verification.json');assert verification['status']=='PASS'and verification['plan_sha256']==sha(EXP/'plan.json')
    best=read(ROOT/'docs/current_best_after_v387_20261106.json');assert sha(ROOT/best['csv'])==best['csv_sha256']==plan['baseline_csv_sha256']
    data=np.load(ROOT/'experiments/regression9_v387/official_rows.npz')
    def arrays(t,p):
        old=dict(np.load(ROOT/'experiments/regression9_v387/controls'/t/(p+'.npz')));current=old['p'].copy();components=[]
        for v in('v386','v387'):
            path=EXP/'components'/v/t/(p+'.npz');z=np.load(path);meta=read(path.with_suffix('.json'));assert sha(path)==meta['prediction_sha256']
            assert meta['config']==plan['fixed_experts'][t][v]and meta['replay_max_error']==0 and meta['query_label_perturbation_pass']
            for k in('fit','query'):np.testing.assert_array_equal(old[k],z[k])
            components.append(z['p'])
        if t=='lipophilicity_astrazeneca':
            prior=np.load(ROOT/'experiments/regression9_v387/outputs/tasks'/t/p/'gaussian_full_C3_r2/predictions.npz')
            current=.9*current+.1*prior['p']
        return old,current,*components
    def score(t,p,a,b):
        old,current,x,z=arrays(t,p);pred=(1-a-b)*current+a*x+b*z;y=data['y'][old['query']]
        base=float(abs(y-current).mean());mae=float(abs(y-pred).mean());value=dict(part=p,baseline=base,candidate=mae,gain=(base-mae)/base)
        if t in TASKS[5:]:value['rank_gain']=float(spearmanr(y,pred).statistic-spearmanr(y,old['score_p']).statistic)
        return value
    dev=read(EXP/'development.json');conf=read(EXP/'confirmation.json');cf={r['task']:r for r in conf['records']};audit=[];accepted=[]
    for r in dev['records']:
        for c in r['choices']:
            assert [score(r['task'],p,c['a'],c['b'])for p in PARTS[:2]]==c['parts']
            objective=min(x['gain']for x in c['parts'])
            if r['task']in TASKS[5:]and min(x['rank_gain']for x in c['parts'])<=0:objective=min(objective,-1.)
            assert objective==c['objective']
        frozen=max(r['choices'],key=lambda x:(x['objective'],-x['a']-x['b'],-x['a']));assert frozen==r['best']
        t=r['task'];c=cf.get(t);parts=[score(t,p,frozen['a'],frozen['b'])for p in PARTS if p in PARTS[:2]or c is not None]
        eligible=len(parts)==4 and all(x['gain']>0 and x.get('rank_gain',1)>0 for x in parts)
        strict=bool(eligible and min(x['gain']for x in parts[:2])>.012 and np.mean([x['gain']for x in parts])>.012 and all(x['lower_positive']for x in c['bootstrap']))
        if c:
            assert parts==c['confirmation']and c['experimental_eligible']==eligible and c['strict_accepted']==strict
            for boot in c['bootstrap']:
                p=boot['part'];old,current,x,z=arrays(t,p);pred=(1-c['a']-c['b'])*current+c['a']*x+c['b']*z
                ix=old['query'];y=data['y'][ix];groups=data['groups'][data['mi'][ix]];unique=np.unique(groups)
                chunks=[np.flatnonzero(groups==g)for g in unique];rng=np.random.default_rng(389000+TASKS.index(t));g=[]
                for _ in range(500):
                    rows=np.concatenate([chunks[j]for j in rng.integers(len(unique),size=len(unique))]);base=abs(y[rows]-current[rows]).mean()
                    g.append(float((base-abs(y[rows]-pred[rows]).mean())/base))
                assert np.quantile(g,[.05,.95]).tolist()==boot['ci90']and len(unique)==boot['groups']
        if eligible:accepted.append(c)
        audit.append(dict(task=t,a=frozen['a'],b=frozen['b'],parts=parts,eligible=eligible,strict_accepted=strict))
    means={p:sum(next(x['gain']for x in r['confirmation']if x['part']==p)for r in accepted)/9 for p in PARTS}
    five={p:sum(next(x['gain']for x in r['confirmation']if x['part']==p)for r in accepted if r['task']in TASKS[:5])/5 for p in PARTS}
    result=dict(version='v389',status='NO_ELIGIBLE_CANDIDATE',baseline=best,official_best=51.314,new_official_score=None,csv=None,
        task_evidence=audit,accepted_tasks=[r['task']for r in accepted],nine_task_mean_relative_MAE_gain_by_part=means,
        five_scored_task_mean_relative_MAE_gain_by_part=five,goal_mean_5percent_achieved_local=all(x>=.05 for x in means.values()),
        strict_stability_confirmed=bool(accepted and all(r['strict_accepted']for r in accepted)),all_fitting_and_weight_selection_remote=True,
        automatic_platform_submission=False,local_bootstrap_recomputed=True,plan_sha256=sha(EXP/'plan.json'),source_sha256=plan['source_sha256'],
        limitations='Repeated public partitions and previously researched fixed experts; historical controls partly proxy; rank MAE calibrated reference. Local gains do not predict official score.')
    if accepted:
        raw=(ROOT/best['csv']).read_bytes();tab=table(raw);spans=field_spans(raw);replacements={};fusion=[]
        for r in accepted:
            t=r['task'];old,current,x,z=arrays(t,'final');pred=(1-r['a']-r['b'])*current+r['a']*x+r['b']*z
            saved=np.load(EXP/(t+'_final.npz'));np.testing.assert_array_equal(pred,saved['p'])
            loc=[i for i,row in enumerate(tab)if i and row[0]==t];original=np.array([float(tab[i][4])for i in loc]);np.testing.assert_allclose(original,current,rtol=0,atol=1e-10)
            assert np.isnan(data['y'][old['query']]).all()and np.isfinite(pred).all()
            assert [tab[i][3]for i in loc]==data['smiles'][data['mi'][old['query']]].tolist()
            for i,value in zip(loc,pred):replacements[i]=format(float(value),'.17g').encode('ascii')
            fusion.append(dict(task=t,baseline_weight=1-r['a']-r['b'],v386_expert_weight=r['a'],v387_expert_weight=r['b'],configs=plan['fixed_experts'][t]))
        chunks=[];cursor=0
        for i,value in sorted(replacements.items()):lo,hi=spans[i][4];chunks.extend([raw[cursor:lo],value]);cursor=hi
        chunks.append(raw[cursor:]);payload=b''.join(chunks);folder=ROOT/'admet_submissions/v389_v387_restricted_expert_fusion_experimental';folder.mkdir(exist_ok=True)
        path=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261106.csv'
        if path.exists():assert path.read_bytes()==payload
        else:path.write_bytes(payload)
        check=verify(path,ROOT/'admet_official/submission_template.csv');changed={}
        for i,(a,b)in enumerate(zip(tab,table(payload))):
            assert a[:4]+a[5:]==b[:4]+b[5:]
            if i not in replacements:assert a==b
            if a!=b:changed[a[0]]=changed.get(a[0],0)+1
        result.update(status='VERIFIED_EXPERIMENTAL_DELIVERY',csv=path.relative_to(ROOT).as_posix(),csv_sha256=sha(path),changed_rows=changed,unchanged_tasks=22-len(accepted),fusion=fusion,format=check)
        save(path.with_suffix('.verification.json'),result)
    save(ROOT/'docs/v389_delivery.json',result)
    print(json.dumps({k:v for k,v in result.items()if k not in('task_evidence','format','baseline')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
