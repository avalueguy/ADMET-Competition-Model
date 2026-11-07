"""Task-specific controls for low-scoring tasks carried into v124."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v117_conditional_structure as old
import v86_metric_ranking as rank86
import v85_smooth_residual as smooth
b,r83=old.b,old.r83
OUT=ROOT/'admet_model_outputs_v125'
ANCHOR=ROOT/'admet_submissions/v124_matched_models_history/我从山北来带着兰花草-小分子药物预测技术创新赛-20261027.csv'
ANCHOR_SHA='a1e175b53c5d5933fd89923c190e4a16c74c116776f5dc1a0abcca771d106faf'
TASKS=('herg','clearance_hepatocyte_az','cyp3a4_substrate_carbonmangels',
       'cyp2d6_substrate_carbonmangels','pgp_broccatelli','solubility_aqsoldb')

def sig():return hashlib.sha256((b.sha(__file__)+old.sig()+ANCHOR_SHA).encode()).hexdigest()
def load():
    d=old.prev.load();assert b.sha(ANCHOR)==ANCHOR_SHA
    d['anchor_rows']=b.table(ANCHOR);return d
def archive(part):
    folder='cv/'+part if part.startswith('fold') else part
    return [np.load(ROOT/f'admet_model_outputs_v67/{folder}/shared/{seed}/predictions.npz') for seed in (672031,672032)]
def split(d,k,part):
    if d['tasks'][k]!='cyp3a4_substrate_carbonmangels':return r83.split(d,k,part)
    z=archive(part)[0];fit=z['fit'][d['tid'][z['fit']]==k];q=z['query'][d['tid'][z['query']]==k]
    assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    assert not set(d['groups'][fit])&set(d['groups'][q]) if part!='final' else True
    return fit,q
def anchor(d,k):return np.array([float(r['y_pred']) for r in d['anchor_rows'] if r['task']==d['tasks'][k]])
def baseline(d,k,part):
    if part=='final':return anchor(d,k)
    task=d['tasks'][k]
    if task=='cyp3a4_substrate_carbonmangels':
        fit,q=split(d,k,part);members=[]
        for z in archive(part):
            np.testing.assert_array_equal(fit,z['fit'][d['tid'][z['fit']]==k])
            pos={int(r):i for i,r in enumerate(z['query'])};members.append(z['p'][[pos[int(i)] for i in q]])
        return np.mean(members,axis=0)
    if task in ('herg','clearance_hepatocyte_az'):
        c=json.loads((smooth.OUT/'confirmation.json').read_text());r=next(x for x in c['records'] if x['task']==task and x['accepted'])
        return smooth.predict(d,k,part,r['config'],r['strength'])
    if task=='cyp2d6_substrate_carbonmangels':
        c=json.loads((rank86.OUT/'confirmation.json').read_text());r=next(x for x in c['records'] if x['task']==task and x['accepted'])
        return rank86.mix(rank86.baseline(d,k,part),rank86.fit(d,k,part,r['config']),r['weight'])
    return r83.baseline(d,k,part)
def features(d,k,part,view):
    fit,q=split(d,k,part);aux,cols,source,_,_=old.assays.assays(d,k,part)
    assert k not in cols and np.all(d['tid'][source]!=k) and np.all(d['split'][source]<(2 if part=='final' else 1))
    raw=d['views']['multiview' if view=='full' else 'physchem']
    x=np.c_[aux[fit],raw[d['mi'][fit]]].astype(np.float32)
    qx=np.c_[aux[q],raw[d['mi'][q]]].astype(np.float32)
    x[~np.isfinite(x)]=np.nan;qx[~np.isfinite(qx)]=np.nan
    return fit,q,x,qx,source
def mix(metric,a,p,w):return (1-w)*a+w*p if metric=='MAE' else (1-w)*r83.rankdata(a)/len(a)+w*r83.rankdata(p)/len(p)
def threshold(metric):return .01 if metric=='MAE' else .008 if metric=='Spearman' else .0025
def main():
    d=load();checks=[]
    for task in TASKS:
        k=d['tasks'].index(task)
        if task=='solubility_aqsoldb':
            checks.append(dict(task=task,control='Approximate historical target proxy; complete v44 OOF unavailable',final_replay=False));continue
        if task=='cyp3a4_substrate_carbonmangels':
            fit,q=split(d,k,'final');ps=[]
            for z in archive('final'):
                pos={int(r):i for i,r in enumerate(z['query'])};ps.append(z['p'][[pos[int(i)] for i in q]])
            p=np.mean(ps,axis=0)
        elif task=='pgp_broccatelli':p=r83.fit(d,k,'final',dict(name='v82_original_d3_t100',view='aux',depth=3,trees=100),(802026,802027,802028),'baseline')
        elif task=='cyp2d6_substrate_carbonmangels':
            c=json.loads((rank86.OUT/'confirmation.json').read_text());r=next(x for x in c['records'] if x['task']==task and x['accepted'])
            p=rank86.mix(rank86.baseline(d,k,'final'),rank86.fit(d,k,'final',r['config']),r['weight'])
        else:
            c=json.loads((smooth.OUT/'confirmation.json').read_text());r=next(x for x in c['records'] if x['task']==task and x['accepted'])
            p=smooth.predict(d,k,'final',r['config'],r['strength'])
        a=anchor(d,k);np.testing.assert_allclose(p,a,atol=1e-12,rtol=0)
        records=[]
        for part in ('fold0','fold1','fold2','valid'):
            fit,q=split(d,k,part);p=baseline(d,k,part)
            records.append(dict(part=part,fit=len(fit),query=len(q),score=r83.score(d['metrics'][k],d['y'][q],p)))
        checks.append(dict(task=task,control='Replayed carried branch; historical upstream OOF remains approximate except CYP3A4 matched v67',final_replay=True,records=records))
    report=dict(status='PASS',signature=sig(),checks=checks,baseline='v124',baseline_score=51.115,
        protected_tasks=[t for t in d['tasks'] if t not in TASKS],test_labels_used=False,
        limitations='Carried branch replay is not full re-OOF of upstream history. Labels reused. Solubility uses proxy; CYP3A4 exact same target fit/query rows.')
    b.save(OUT/'baseline_audit.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
