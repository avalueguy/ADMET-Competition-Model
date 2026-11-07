"""Remote-only restricted combination of two previously frozen task experts."""
from pathlib import Path
import hashlib,json,os,platform,sys,time,tarfile
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent
TASKS=('caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb','clearance_hepatocyte_az','clearance_microsome_az','half_life_obach','vdss_lombardo')
PARTS=('fold0','fold1','fold2','valid')
GRID=(0.,.05,.1,.2,.35,.5,.75)

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def modules(v):
    # Each process imports only one legacy module graph; no cross-root globals.
    path=ROOT.parent/f'regression9_{v}';sys.path.insert(0,str(path))
    return __import__('v386_oof_calibrated'if v=='v386'else'v387_correlated_metric')

def worker(v,t,p,c):
    m=modules(v);m.setup();d=m.load();pred=m.predict(d,t,p,c);rp=m.replay(d,t,p,c)
    np.testing.assert_array_equal(pred,rp);a=m.design(d,t,p)
    changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.;np.testing.assert_array_equal(rp,m.replay(changed,t,p,c))
    dest=ROOT/'components'/v/t/(p+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,fit=a['fit'],query=a['query'],p=rp,signature=m.signature())
    return dict(source=v,task=t,part=p,config=c,prediction_sha256=sha(dest),source_signature=m.signature(),replay_max_error=0.,query_label_perturbation_pass=True)

def components(plan,t,p):
    import subprocess
    for v in('v386','v387'):
        dest=ROOT/'components'/v/t/(p+'.npz')
        if not dest.exists():
            c=plan['fixed_experts'][t][v]
            r=subprocess.run([sys.executable,str(Path(__file__)),v,t,p,json.dumps(c)],cwd=ROOT,capture_output=True,text=True)
            assert r.returncode==0,r.stderr
            save(dest.with_suffix('.json'),json.loads(r.stdout.splitlines()[-1]))
    z=[dict(np.load(ROOT/'components'/v/t/(p+'.npz')))for v in('v386','v387')]
    old=dict(np.load(ROOT.parent/'regression9_v387/controls'/t/(p+'.npz')))
    for a in z:
        for k in('fit','query'):np.testing.assert_array_equal(old[k],a[k])
    current=old['p'].copy()
    if t=='lipophilicity_astrazeneca':
        incumbent=np.load(ROOT.parent/'regression9_v387/outputs/tasks'/t/p/'gaussian_full_C3_r2/predictions.npz')
        current=.9*current+.1*incumbent['p']
    return old,current,z[0]['p'],z[1]['p']

def score(d,plan,t,p,a,b):
    old,current,x,z=components(plan,t,p);pred=(1-a-b)*current+a*x+b*z;y=d['y'][old['query']]
    base=float(abs(y-current).mean());mae=float(abs(y-pred).mean())
    value=dict(part=p,baseline=base,candidate=mae,gain=(base-mae)/base)
    if t in TASKS[5:]:value['rank_gain']=float(spearmanr(y,pred).statistic-spearmanr(y,old['score_p']).statistic)
    return value

def bootstrap(d,plan,t,p,a,b):
    old,current,x,z=components(plan,t,p);pred=(1-a-b)*current+a*x+b*z;ix=old['query'];y=d['y'][ix]
    groups=d['groups'][d['mi'][ix]];unique=np.unique(groups);chunks=[np.flatnonzero(groups==g)for g in unique];rng=np.random.default_rng(389000+TASKS.index(t));g=[]
    for _ in range(500):
        rows=np.concatenate([chunks[j]for j in rng.integers(len(unique),size=len(unique))]);base=abs(y[rows]-current[rows]).mean()
        g.append(float((base-abs(y[rows]-pred[rows]).mean())/base))
    ci=np.quantile(g,[.05,.95]).tolist();return dict(part=p,ci90=ci,lower_positive=ci[0]>0,groups=len(unique))

def main():
    import torch
    assert platform.system()=='Linux'and torch.cuda.is_available()
    plan=read(ROOT/'plan.json');assert sha(Path(__file__))==plan['source_sha256']
    assert not(ROOT/'launch.json').exists();start=time.time();save(ROOT/'launch.json',dict(pid=os.getpid(),at_unix=start))
    d=dict(np.load(ROOT.parent/'regression9_v387/official_rows.npz'));records=[];selected=[]
    weights=[(a,b)for a in GRID for b in GRID if 0<a+b<=.75]
    save(ROOT/'run_status.json',dict(stage='DEVELOPMENT',pid=os.getpid(),at_unix=time.time()))
    for t in TASKS:
        choices=[]
        for a,b in weights:
            parts=[score(d,plan,t,p,a,b)for p in PARTS[:2]];obj=min(x['gain']for x in parts)
            if t in TASKS[5:]and min(x['rank_gain']for x in parts)<=0:obj=min(obj,-1.)
            choices.append(dict(task=t,a=a,b=b,parts=parts,objective=obj))
        best=max(choices,key=lambda x:(x['objective'],-x['a']-x['b'],-x['a']))
        records.append(dict(task=t,best=best,choices=choices))
        if best['objective']>0:selected.append(best)
    save(ROOT/'development.json',dict(records=records,weights=weights));save(ROOT/'frozen_selection.json',dict(selected=selected,no_fallback=True))
    save(ROOT/'run_status.json',dict(stage='CONFIRMATION',pid=os.getpid(),at_unix=time.time()))
    confirmed=[]
    for r in selected:
        t,a,b=r['task'],r['a'],r['b'];parts=[score(d,plan,t,p,a,b)for p in PARTS];boots=[bootstrap(d,plan,t,p,a,b)for p in PARTS[2:]]
        eligible=all(x['gain']>0 and x.get('rank_gain',1)>0 for x in parts)
        strict=eligible and min(x['gain']for x in parts[:2])>.012 and np.mean([x['gain']for x in parts])>.012 and all(x['lower_positive']for x in boots)
        value=dict(**r,confirmation=parts,bootstrap=boots,experimental_eligible=bool(eligible),strict_accepted=bool(strict));confirmed.append(value)
        if eligible:
            old,current,x,z=components(plan,t,'final');pred=(1-a-b)*current+a*x+b*z;assert np.isnan(d['y'][old['query']]).all()
            np.savez_compressed(ROOT/(t+'_final.npz'),fit=old['fit'],query=old['query'],p=pred)
    save(ROOT/'confirmation.json',dict(records=confirmed))
    # Recompute fixed selection and every confirmation from immutable components.
    for r in records:
        for c in r['choices']:assert [score(d,plan,r['task'],p,c['a'],c['b'])for p in PARTS[:2]]==c['parts']
    for r in confirmed:assert [score(d,plan,r['task'],p,r['a'],r['b'])for p in PARTS]==r['confirmation']
    value=dict(status='PASS',selection_recomputed=True,components_replayed=True,query_label_perturbation_pass=True,
        source_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'plan.json'),new_training_fits=False,
        source_prediction_generation_may_fit_previously_unrequested_confirmation_or_final_parts=True)
    save(ROOT/'verification.json',value)
    save(ROOT/'run_status.json',dict(stage='EVIDENCE_READY',pid=os.getpid(),at_unix=time.time(),seconds=time.time()-start))
    files=sorted(p for p in ROOT.rglob('*')if p.is_file()and p.name not in('results.tar.gz','result_archive.json','training.log'))
    save(ROOT/'result_manifest.json',dict(files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))for p in files]))
    with tarfile.open(ROOT/'results.tar.gz','w:gz')as tf:
        for p in files+[ROOT/'result_manifest.json']:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
    save(ROOT/'result_archive.json',dict(status='PASS',sha256=sha(ROOT/'results.tar.gz'),bytes=(ROOT/'results.tar.gz').stat().st_size))
    print(json.dumps(dict(eligible=[r['task']for r in confirmed if r['experimental_eligible']],seconds=time.time()-start)))

if __name__=='__main__':
    if len(sys.argv)>1:print(json.dumps(worker(sys.argv[1],sys.argv[2],sys.argv[3],json.loads(sys.argv[4]))))
    else:main()
