"""Refit only frozen accepted MAE tasks, reload, and write exact-template CSV."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v74_specialists as v
from verify_submission import verify
from repack_upload_template import field_spans

def reload_neural(d,g,k,folder,probe=False):
    z=np.load(folder/'predictions.npz');fit=z['fit'];q=z['query'];assert str(z['signature'])==v.signature();b.assert_excluded(d,fit,q)
    x,state=v.design(d,g,fit);ps=[];records=[]
    for j in range(3):
        file=folder/f'head{j}.pt';ck=torch.load(file,map_location='cpu',weights_only=False);assert ck['signature']==v.signature() and ck['head']==j
        for key in ('prior_mean','prior_scale'):np.testing.assert_array_equal(state[key],ck['design'][key])
        for key,value in state['pre'].items():np.testing.assert_array_equal(value,ck['design']['pre'][key])
        net=v.Net(ck['fdim'],ck['task']).cuda();net.load_state_dict(ck['state']);p=v.predict(net,d,g,x,q,k,ck['center'],ck['scale'])[j]
        np.testing.assert_allclose(p,z['p'][j],atol=3e-5,rtol=3e-6)
        if probe:
            ii=np.unique([len(q)-1,0,len(q)//2]);qq=q[ii]
            tiny=v.predict(net,d,g,x,qq,k,ck['center'],ck['scale'],batch=1)[j]
            np.testing.assert_allclose(tiny,p[ii],atol=3e-5,rtol=3e-6)
        ps.append(p);records.append(dict(path=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),max_error=float(np.max(abs(p-z['p'][j]))),batch_probe=probe));del net
    del x;torch.cuda.empty_cache();return np.stack(ps),records

def main():
    assert torch.cuda.is_available() and v.os.name!='nt';torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    d=b.build();g=v.graphs(d);confirmation=json.loads((v.OUT/'confirmation.json').read_text(encoding='utf-8'));assert confirmation['signature']==v.signature()
    accepted=[r for r in confirmation['tasks'] if r['accepted']];assert accepted,'No task passed; do not manufacture a release'
    selected={};checks=[]
    for r in accepted:
        task=r['task'];k=d['tasks'].index(task);c=r['selection'];fit,q=v.rows(d,k,'final')
        epochs=np.median([json.loads((v.OUT/task/part/'inner/metadata.json').read_text(encoding='utf-8'))['epochs'] for part in v.PARTS],axis=0).astype(int).clip(1).tolist()
        trees=int(np.median([json.loads((v.OUT/task/part/'tree_inner/metadata.json').read_text(encoding='utf-8'))['trees'] for part in v.PARTS]))
        parts=[]
        for seed in v.P['final_seeds']:
            folder=v.OUT/task/'final'/str(seed);pp=v.train_net(d,g,k,fit,q,seed,folder,epochs);_,a=reload_neural(d,g,k,folder,True);parts.append(pp);checks.extend(a)
        expert=np.mean(parts,axis=0)[c['heads']].mean(0)
        if c['tree_weight']:
            tree=v.train_tree(d,k,fit,q,v.OUT/task/'final/tree',trees);expert=(1-c['tree_weight'])*expert+c['tree_weight']*tree
        selected[task]=dict(choice=c,epochs=epochs,trees=trees,p=expert)
    base=next((ROOT/'admet_submissions/v73_v72').glob('*.csv'));assert b.sha(base)==v.P['anchor_sha256']
    with base.open(encoding='utf-8-sig',newline='') as f:old=list(csv.reader(f))
    new=[r[:] for r in old];replay={}
    for task,rec in selected.items():
        pos=[i for i,r in enumerate(new) if i and r[0]==task];a=np.array([float(new[i][4]) for i in pos]);p=rec['p'];assert len(p)==len(pos)
        w=rec['choice']['anchor_weight'];combined=(1-w)*a+w*p
        for i,value in zip(pos,combined):new[i][4]=repr(float(value))
        replay[task]=dict(anchor=a.tolist(),expert=p.tolist(),result=combined.tolist(),weight=w,changed_rows=sum(new[i][4]!=old[i][4] for i in pos))
    template=ROOT/'admet_official/submission_template.csv';ref=template.read_bytes();spans=field_spans(ref);chunks=[];cursor=0
    for i,r in enumerate(new[1:],1):
        lo,hi=spans[i][4];chunks.extend((ref[cursor:lo],r[4].encode('ascii')));cursor=hi
    chunks.append(ref[cursor:]);raw=b''.join(chunks);actual=field_spans(raw);chunks=[];cursor=0
    for i in range(1,len(new)):
        lo,hi=actual[i][4];a,z=spans[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    from datetime import datetime,timezone,timedelta
    day=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d');dest=ROOT/'admet_submissions/v74_v73';dest.mkdir(parents=True,exist_ok=True)
    file=dest/f'我从山北来带着兰花草-小分子药物预测技术创新赛-{day}.csv'
    if file.exists():assert file.read_bytes()==raw
    else:file.write_bytes(raw)
    audit=verify(file,template);audit.update(nonprediction_bytes_exact=True,changed_tasks={t:r['changed_rows'] for t,r in replay.items()},independent_process_verification_pending=True)
    b.save(file.with_suffix('.verification.json'),audit);b.save(v.OUT/'fusion_replay.json',replay)
    b.save(v.OUT/'release.json',dict(status='CANDIDATE_PENDING_INDEPENDENT_PROCESS_VERIFY',signature=v.signature(),anchor='v73',anchor_sha256=b.sha(base),anchor_official_score_user_reported=50.640,
        file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),selected={t:{key:value for key,value in r.items() if key!='p'} for t,r in selected.items()},reload_checks=checks,
        test_labels_used=False,new_branch_external_data=False,new_branch_pretrained_weights=False,anchor_contains_historical_pretraining=True,official_score=None,
        limitations='Full v73 OOF unavailable. New same-fit-row tree controls and separately frozen confirmations improve comparison discipline but use historically reused labels; proxy comparisons do not prove platform gains.'))
    print(json.dumps(dict(status='EXPORTED',file=str(file),changed_tasks=list(selected),sha256=b.sha(file)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
