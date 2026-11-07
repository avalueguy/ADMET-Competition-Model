"""Refit frozen accepted v75 branches and insert predictions into exact template."""
from pathlib import Path
import sys,json,csv
from datetime import datetime,timedelta,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v75_experiments as v
from repack_upload_template import field_spans
from verify_submission import verify,read


def main():
    confirmation=json.loads((v.OUT/'confirmation.json').read_text());assert confirmation['signature']==v.signature()
    assert b.sha(v.OUT/'frozen_selection.json')==confirmation['frozen_sha256']
    accepted=[r for r in confirmation['tasks'] if r['accepted']];assert accepted
    d=b.build();views=None;selected={};replay={}
    base=next((ROOT/'admet_submissions/v73_v72').glob('*.csv'));assert b.sha(base)==v.P['anchor_sha256']
    old=read(base);new=[r[:] for r in old]
    for r in accepted:
        task=r['task'];k=d['tasks'].index(task);metric=d['metrics'][k];choice=r['selection'];fit,q=v.rows(d,k,'final')
        conditional=None;trees=None
        if choice['conditional']:
            trees=max(1,int(np.median([json.loads((v.OUT/task/part/'conditional_tree/inner/metadata.json').read_text())['trees'] for part in v.P['parts']])))
            aux,audit=v.auxiliaries(d,k,True)
            pp=[v.fit_tree(d,k,fit,q,v.OUT/task/'final'/f'conditional_{seed}',aux,audit,trees,seed) for seed in v.P['final_seeds']]
            conditional=np.mean(pp,axis=0)
        expert=conditional
        if choice['kernel'] is not None:
            if views is None:views=v.fingerprints(d)
            raw=v.fit_kernels(d,views,k,'final')[choice['kernel']]
            expert=raw if conditional is None else b.blend(metric,raw,conditional,choice['conditional'])
        pos=[i for i,r in enumerate(new) if i and r[0]==task];a=np.array([float(new[i][4]) for i in pos])
        assert len(expert)==len(pos)
        combined=b.blend(metric,a,expert,v.P['anchor_weight'])
        if task=='ppbr_az':combined=combined.clip(0,100)
        for i,value in zip(pos,combined):new[i][4]=repr(float(value))
        selected[task]=dict(choice=choice,trees=trees,metric=metric)
        replay[task]=dict(anchor=a.tolist(),expert=expert.tolist(),result=combined.tolist(),weight=v.P['anchor_weight'],
            metric=metric,changed_rows=sum(new[i][4]!=old[i][4] for i in pos))
    template=ROOT/'admet_official/submission_template.csv';ref=template.read_bytes();spans=field_spans(ref);chunks=[];cursor=0
    for i,row in enumerate(new[1:],1):
        lo,hi=spans[i][4];chunks.extend((ref[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(ref[cursor:]);raw=b''.join(chunks)
    day=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d');dest=ROOT/'admet_submissions/v75_v73';dest.mkdir(parents=True,exist_ok=True)
    file=dest/f'我从山北来带着兰花草-小分子药物预测技术创新赛-{day}.csv'
    if file.exists():assert file.read_bytes()==raw
    else:file.write_bytes(raw)
    audit=verify(file,template);b.save(file.with_suffix('.verification.json'),audit);b.save(v.OUT/'fusion_replay.json',replay)
    b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_PROCESS_VERIFY',signature=v.signature(),file=file.relative_to(ROOT).as_posix(),
        sha256=b.sha(file),anchor='v73',anchor_sha256=b.sha(base),anchor_official_score_user_reported=50.640,selected=selected,
        official_score=None,test_labels_used=False,new_branch_external_data=False,new_branch_pretrained_weights=False,
        anchor_contains_historical_pretraining=True,exact_full_anchor_oof=False,limitations=v.P['limitations']))
    print(json.dumps(dict(status='EXPORTED',file=str(file),changed_tasks=list(selected),sha256=b.sha(file)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
