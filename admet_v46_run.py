"""Remote task adaptation with explicitly replayed, scored submission baseline."""
import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit
import admet_v24_models as b
import admet_v27_models as graph
import admet_v44_run as v44

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'admet_model_outputs_v46'
P = json.loads((ROOT/'admet_v46_protocol.json').read_text())
FEATURES = {}

def signature():
    names = ['admet_v46_run.py','admet_v46_protocol.json','admet_model_outputs_v46/v45_source_export.json']
    return hashlib.sha256(json.dumps(dict(graph=graph.signature(),v44=v44.signature(),
        files={x:b.sha(ROOT/x) for x in names}),sort_keys=True).encode()).hexdigest()

def original(d,k,partition):
    parts = []
    expected = None
    for seed in P['source_seeds']:
        z = np.load(graph.OUT/partition/str(seed)/'predictions.npz')
        assert str(z['signature']) == graph.signature()
        q = z['query'].astype(int)
        active = d['tid'][q] == k
        if expected is None:expected = q[active]
        np.testing.assert_array_equal(expected,q[active])
        parts.append(z['p'][:,active])
    return expected,np.mean(parts,axis=0)

def baseline(d):
    # v44 cached baseline arrays represent v43, not the v44 submission.
    valid = np.load(v44.OUT/'baseline_valid.npy').copy()
    cv = np.load(v44.OUT/'baseline_cv_proxy.npy').copy()
    folds = np.load(b.OUT/'folds.npy')
    report = json.loads((v44.OUT/'analysis.json').read_text())
    assert report['signature'] == v44.signature()
    for r in report['tasks']:
        s = r['selection']
        if not s:continue
        k = d['tasks'].index(r['task'])
        vi = np.flatnonzero((d['tid']==k)&(d['split']==1))
        tr = np.flatnonzero((d['tid']==k)&(d['split']==0))
        z = np.load(v44.OUT/r['task']/'valid/predictions.npz')
        vp = v44.variants(z['p'],r['metric'])[s['variant']]
        valid[vi] = b.blend(r['metric'],valid[vi],vp,s['weight'])
        cp = np.load(v44.OUT/r['task']/'cv_predictions.npy')[s['variant']]
        for fold in range(3):
            ix = tr[folds[tr]==fold]
            cv[ix] = b.blend(r['metric'],cv[ix],cp[ix],s['weight'])
    exported = json.loads((OUT/'v45_source_export.json').read_text())
    assert exported['sha256'] == P['baseline_sha256']
    for task,s in exported['changed_tasks'].items():
        k = d['tasks'].index(task)
        m = d['metrics'][k]
        vi = np.flatnonzero((d['tid']==k)&(d['split']==1))
        tr = np.flatnonzero((d['tid']==k)&(d['split']==0))
        vp = np.load(graph.OUT/'valid_predictions.npy')[s['branch'],vi]
        cp = np.load(graph.OUT/'cv_predictions.npy')[s['branch'],tr]
        valid[vi] = b.blend(m,valid[vi],vp,s['weight'])
        for fold in range(3):
            active = folds[tr]==fold
            ix = tr[active]
            cv[ix] = b.blend(m,cv[ix],cp[active],s['weight'])
    np.save(OUT/'baseline_valid.npy',valid)
    np.save(OUT/'baseline_cv_proxy.npy',cv)
    b.save(OUT/'baseline_replay.json',dict(signature=signature(),v44_changes_replayed=True,
        v45_export_sha256=exported['sha256'],v45_changes_replayed=True,
        historical_cv_proxy=True,official_score_prediction=None))
    return valid,cv

def source_partition(name):
    return 'cv/'+name if name.startswith('fold') else name

def feature_matrix(d,ck,partition):
    # Both v27 seeds use the same fit-only preprocessor for a given partition.
    if partition not in FEATURES:
        raw = np.concatenate([d['desc'],d['emb']],1)
        pre = ck['preprocessor']
        x = np.concatenate([d['fp2'],d['fcfp'],pre.transform(raw)],1)
        FEATURES.clear()
        FEATURES[partition] = torch.tensor(x,device='cuda')
    return FEATURES[partition]

def logits(net,g,x,molecules):
    net.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0,len(molecules),96):
            mm = molecules[start:start+96]
            parts.append(net(graph.pack(g,mm),x[mm]).cpu().numpy())
    return np.concatenate(parts,axis=1)

def fit_seed(d,g,k,rows,query,seed,dest,partition):
    dest.mkdir(parents=True,exist_ok=True)
    cache = dest/'predictions.npz'
    sig = signature()
    if cache.exists():
        z = np.load(cache)
        assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],rows)
        np.testing.assert_array_equal(z['query'],query)
        return z['p'].copy()
    assert np.all(d['split'][rows]<2) and not set(rows)&set(query)
    if partition!='final':
        assert not set(d['rg'][rows])&set(d['rg'][query])
        assert not set(d['canonical'][d['mi'][rows]])&set(d['canonical'][d['mi'][query]])
    source = graph.OUT/partition/str(seed)
    source_prediction = np.load(source/'predictions.npz')
    source_fit = source_prediction['fit']
    if partition!='final':
        assert not set(d['rg'][source_fit])&set(d['rg'][query])
    ck = torch.load(source/f'checkpoint_{P["source_step"]}.pt',map_location='cpu',weights_only=False)
    assert ck['signature']==graph.signature()
    torch.manual_seed(seed+46000+k)
    torch.cuda.manual_seed_all(seed+46000+k)
    rng = np.random.default_rng(seed+46000+k)
    net = graph.BondNet(*ck['dimensions']).cuda()
    net.load_state_dict(ck['state'])
    x = feature_matrix(d,ck,partition)
    fit_mol = np.unique(d['mi'][rows])
    y = np.array([np.mean(d['y'][rows[d['mi'][rows]==mi]]) for mi in fit_mol],np.float32)
    teacher = logits(net,g,x,fit_mol)[:,:,k]
    targets = torch.tensor(y,device='cuda')
    teacher = torch.tensor(teacher,device='cuda')
    qp = np.asarray(source_prediction['query'],int)
    pos = np.searchsorted(qp,query)
    np.testing.assert_array_equal(qp[pos],query)
    original_p = source_prediction['p'][:,pos]
    heads = list(net.graph_heads.parameters())+list(net.hybrid_heads.parameters())
    head_ids = {id(x) for x in heads}
    encoder = [x for x in net.parameters() if id(x) not in head_ids]
    optimizer = torch.optim.AdamW([
        {'params':encoder,'lr':P['encoder_learning_rate']},
        {'params':heads,'lr':P['head_learning_rate']}],weight_decay=P['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,P['steps'],eta_min=1e-5)
    snaps = []
    t0 = time.time()
    for step in range(1,P['steps']+1):
        net.train()
        ix = rng.choice(len(fit_mol),size=min(P['batch_size'],len(fit_mol)),replace=True)
        mm = fit_mol[ix]
        pred = net(graph.pack(g,mm),x[mm])[:,:,k]
        true = targets[ix]
        loss = nn.functional.binary_cross_entropy_with_logits(pred,true.expand_as(pred))
        positive,negative = pred[:,true>.5],pred[:,true<.5]
        if positive.shape[1] and negative.shape[1]:
            loss = loss+P['ranking_coefficient']*nn.functional.softplus(
                negative[:,None,:]-positive[:,:,None]).mean()
        loss = loss+P['distillation_coefficient']*nn.functional.mse_loss(pred,teacher[:,ix])
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),3.)
        optimizer.step()
        schedule.step()
        if step in P['snapshot_steps']:
            p = expit(logits(net,g,x,d['mi'][query])[:,:,k])
            snaps.append(p)
            saved = dict(state={a:c.detach().cpu().clone() for a,c in net.state_dict().items()},
                dimensions=ck['dimensions'],preprocessor=ck['preprocessor'],target_task=k,
                source_checkpoint=str(source/f'checkpoint_{P["source_step"]}.pt'),
                source_sha256=b.sha(source/f'checkpoint_{P["source_step"]}.pt'),step=step,signature=sig)
            path = dest/f'checkpoint_{step}.pt'
            torch.save(saved,path)
            restored = torch.load(path,map_location='cpu',weights_only=False)
            net.load_state_dict(restored['state'])
            np.testing.assert_allclose(p,expit(logits(net,g,x,d['mi'][query])[:,:,k]),rtol=3e-5,atol=3e-5)
        if step%100==0:
            b.emit('v46_adapt',task=d['tasks'][k],partition=partition,seed=seed,step=step,
                loss=float(loss.detach()),seconds=round(time.time()-t0))
    p = np.concatenate([original_p,np.mean(snaps,axis=0)])
    np.savez_compressed(cache,p=p,fit=rows,query=query,signature=sig)
    b.save(dest/'metadata.json',dict(signature=sig,task=d['tasks'][k],partition=partition,
        fit_rows=len(rows),query_rows=len(query),source_fit_rows=len(source_fit),
        source_query_scaffold_overlap=len(set(d['rg'][source_fit])&set(d['rg'][query])),
        fine_tuning_query_scaffold_overlap=len(set(d['rg'][rows])&set(d['rg'][query])),
        source_signature=graph.signature(),model_reload_passed=True,seconds=time.time()-t0,
        test_labels_used=False,external_labels=False,local_gpu_training=False))
    del net,optimizer
    torch.cuda.empty_cache()
    return p

def fit(d,g,k,rows,query,dest):
    partition = source_partition(dest.name)
    return np.mean([fit_seed(d,g,k,rows,query,seed,dest/str(seed),partition)
        for seed in P['source_seeds']],axis=0)

def variants(p):
    p = np.stack([b.rank(x) for x in p])
    return np.concatenate([p,np.mean(p[2:],axis=0)[None],np.mean(p,axis=0)[None]])

def evaluate(d,g):
    base,old = baseline(d)
    folds = np.load(b.OUT/'folds.npy')
    records = []
    for task in P['tasks']:
        k = d['tasks'].index(task)
        m = d['metrics'][k]
        assert m in ['AUROC','AUPRC']
        tr = np.flatnonzero((d['tid']==k)&(d['split']==0))
        vi = np.flatnonzero((d['tid']==k)&(d['split']==1))
        vp = variants(fit(d,g,k,tr,vi,OUT/task/'valid'))
        cp = np.full((6,len(d['y'])),np.nan)
        for fold in range(3):
            q = tr[folds[tr]==fold]
            rows = tr[folds[tr]!=fold]
            cp[:,q] = variants(fit(d,g,k,rows,q,OUT/task/f'fold{fold}'))
        np.save(OUT/task/'valid_predictions.npy',vp)
        np.save(OUT/task/'cv_predictions.npy',cp)
        grid = []
        for v in range(6):
            for w in P['weights']:
                vg = b.delta(m,d['y'][vi],base[vi],b.blend(m,base[vi],vp[v],w))
                gs = [b.delta(m,d['y'][ix],old[ix],b.blend(m,old[ix],cp[v,ix],w))
                    for fold in range(3) for ix in [tr[folds[tr]==fold]]]
                obj = float(.20*vg+.80*np.mean(gs)-.05*np.std(gs))
                grid.append(dict(variant=v,name=P['variants'][v],weight=w,valid_gain=vg,
                    cv_gains=gs,cv_mean=float(np.mean(gs)),objective=obj,
                    eligible=bool(np.mean(gs)>.0005 and obj>.0005 and sum(x>0 for x in gs)>=2)))
        choices = [r for r in grid if r['eligible']]
        s = max(choices,key=lambda r:r['objective']) if choices else None
        records.append(dict(task=task,metric=m,selection=s,grid=grid,
            baseline_valid_metric=b.metric(m,d['y'][vi],base[vi])))
        b.save(OUT/'analysis.json',dict(signature=signature(),status='IN_PROGRESS',tasks=records))
        b.emit('v46_selection',task=task,selection=s)
    b.save(OUT/'analysis.json',dict(signature=signature(),status='FROZEN',baseline='v45',
        baseline_score=P['baseline_score'],tasks=records,selected_tasks=[r['task'] for r in records if r['selection']],
        official_score_prediction=None,limitations=P['limitations']))

def read_csv(path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))

def final_export(d,g):
    report = json.loads((OUT/'analysis.json').read_text())
    assert report['signature']==signature() and report['status']=='FROZEN'
    source = OUT/'baseline_v45.csv'
    assert b.sha(source)==P['baseline_sha256']
    rows = read_csv(source)
    template = read_csv(ROOT/'admet_official/submission_template.csv')
    changed,compact,replay = {},{},{}
    for r in report['tasks']:
        s = r['selection']
        if not s:continue
        k = d['tasks'].index(r['task'])
        fitrows = np.flatnonzero((d['tid']==k)&(d['split']<2))
        query = np.flatnonzero((d['tid']==k)&(d['split']==2))
        p = fit(d,g,k,fitrows,query,OUT/r['task']/'final')
        compact[r['task']] = p.tolist()
        pos = [i for i,a in enumerate(rows) if i>0 and a[0]==r['task']]
        old = np.array([float(rows[i][4]) for i in pos])
        new = b.blend(r['metric'],old,variants(p)[s['variant']],s['weight'])
        for i,value in zip(pos,new):rows[i][4]=repr(float(value))
        changed[r['task']] = dict(**s,rows=int(np.sum(old!=new)))
        replay[r['task']] = dict(metric=r['metric'],weight=s['weight'],variant=s['variant'],
            baseline=old.tolist(),source=p.tolist(),result=new.tolist())
        b.save(OUT/'test_predictions_compact.json',compact)
    assert changed and len(rows)==len(template)==16380 and rows[0]==template[0]
    for row,trow in zip(rows[1:],template[1:]):
        assert len(row)==6 and row[:4]+row[5:]==trow[:4]+trow[5:]
        assert np.isfinite(float(row[4]))
        if row[5] in ['AUROC','AUPRC']:assert 0<=float(row[4])<=1
    dest = ROOT/'admet_submissions/v46'
    dest.mkdir(parents=True,exist_ok=True)
    file = dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261015-v46.csv'
    buf = io.StringIO(newline='')
    csv.writer(buf,lineterminator='\r\n').writerows(rows)
    file.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'))
    assert read_csv(file)==rows
    b.save(OUT/'fusion_replay.json',replay)
    b.save(dest/'verification.json',dict(status='EXPERIMENTAL_AWAITING_PLATFORM',signature=signature(),
        baseline='v45',baseline_score=P['baseline_score'],baseline_sha256=P['baseline_sha256'],
        sha256=b.sha(file),rows=16379,tasks=22,columns=6,changed_tasks=changed,
        bytes=file.stat().st_size,encoding='UTF-8 BOM',line_endings='CRLF',template_identity_exact=True,
        model_reload_verified=True,local_gpu_training=False,automatically_uploaded=False,
        platform_score=None,limitations=P['limitations']))
    b.emit('v46_export',sha256=b.sha(file),changed_tasks=list(changed))

def main():
    a = argparse.ArgumentParser()
    a.add_argument('stage',choices=['all','evaluate','final'])
    args = a.parse_args()
    b.remote_only()
    assert ROOT==Path(P['remote_root'])
    torch.backends.cuda.matmul.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=True)
    d = b.build_data()
    g = graph.graphs(d)
    if args.stage in ['all','evaluate']:evaluate(d,g)
    if args.stage in ['all','final']:final_export(d,g)
    b.save(OUT/(args.stage+'_completed.json'),dict(signature=signature(),completed=True))

if __name__=='__main__':main()
