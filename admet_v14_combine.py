"""Combine verified v12/v13 task routes; no fitting or weight search.

Both independent experiments use the same scored v11. Use disjoint promotions.
On overlaps, v13 replaces v12 only if it passes the original nonregression gates
against v12 itself. Otherwise v12 is retained. No individual molecule routing.
"""
import csv,io,json,shutil,hashlib,tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from admet_v6_runner import remote_only,metrics,kind,deltas,secondary_ok,get_metric,save,sha
from admet_v12_models import load_task
from admet_v10_cumulative_audit import evaluate_split

ROOT=Path('admet_release_v14')
BASE=Path('admet_release_v11')
NAME='我从山北来带着兰花草-小分子药物预测技术创新赛-20260925.csv'
OLD=Path('admet_submissions/v11')/NAME
UPLOAD=Path('admet_submissions/v14')/NAME


def read(path):return json.loads(Path(path).read_text())


def predictions(version,task):
    root=Path(f'admet_model_outputs_v{version}')/task;s=read(root/'decision.json')['selection']
    b=pd.read_csv(root/'baseline_oof.csv');c=pd.read_csv(root/(s['family']+'_oof.csv'))
    np.testing.assert_array_equal(b.fold,c.fold);np.testing.assert_allclose(b.Y_true,c.Y_true)
    b['new']=(1-s['alpha'])*b.Y_pred+s['alpha']*c.Y_pred
    return b,pd.read_csv(root/'validation_comparison.csv'),pd.read_csv(root/'repeat_comparison.csv')


def checks_for(task,old,new):
    d=load_task(task);metric=get_metric(task);n=len(d['frames']['train']);nv=len(d['frames']['valid']);rows=[]
    bo,vo,ro=old;bn,vn,rn=new
    np.testing.assert_array_equal(bo.fold,bn.fold);np.testing.assert_allclose(bo.Y_true,bn.Y_true)
    for f in range(3):
        it=np.flatnonzero(bo.fold!=f);iv=np.flatnonzero(bo.fold==f)
        rows+=evaluate_split(task,f'fold_{f}',d,it,iv,bo.Y_true.to_numpy()[iv],bo.new.to_numpy()[iv],bn.new.to_numpy()[iv],metric)
    np.testing.assert_allclose(vo.Y_true,vn.Y_true);np.testing.assert_array_equal(ro.row,rn.row);np.testing.assert_allclose(ro.Y_true,rn.Y_true)
    rows+=evaluate_split(task,'valid',d,np.arange(n),np.arange(n,n+nv),vo.Y_true.to_numpy(),vo.blend.to_numpy(),vn.blend.to_numpy(),metric)
    it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=692509).split(np.arange(n),groups=d['groups'][:n]))
    np.testing.assert_array_equal(rn.row,iv)
    rows+=evaluate_split(task,'repeat',d,it,iv,ro.Y_true.to_numpy(),ro.blend.to_numpy(),rn.blend.to_numpy(),metric)
    return rows


def main():
    remote_only();ROOT.mkdir(exist_ok=True)
    assert sha(OLD)=='082183f0c462081ea8735a41ec2e47326ef4cdcaf9d1188ae86a006f8057122d'
    verified={v:read(f'admet_release_v{v}/independent_verification.json') for v in [12,13]}
    for v,x in verified.items():
        assert x['status']=='VERIFIED' and x['independent_replay']
        assert read(f'admet_diagnostics/v{v}_anchor_audit.json')['status']=='PASSED'
        assert sha(x['submission'])==x['sha256']
    tasks=sorted(p.name for p in BASE.iterdir() if p.is_dir());routes=[];all_checks=[];overlap_checks=[]
    for task in tasks:
        available=[v for v in [12,13] if task in verified[v]['accepted']]
        reason='retain scored v11';chosen=11
        if len(available)==1:chosen=available[0];reason='sole accepted independent candidate'
        if len(available)==2:
            comparison=checks_for(task,predictions(12,task),predictions(13,task));overlap_checks+=comparison
            safe=all(r['coverage_sufficient'] and r['primary_pass'] and r['secondary_pass'] for r in comparison)
            chosen=13 if safe else 12;reason='v13 passes against v12' if safe else 'retain v12 after overlap gate'
        source=Path(f'admet_release_v{chosen}')/task;dest=ROOT/task;dest.mkdir(exist_ok=True)
        for name in [f'submission_{task}.csv','validation_predictions.csv']:shutil.copy2(source/name,dest/name)
        route=dict(task=task,version=chosen,changed=chosen!=11,reason=reason,prediction_sha256=sha(dest/f'submission_{task}.csv'),source_selection_sha256=sha(source/'selection.json'))
        save(dest/'selection.json',route);routes.append(route)
        if chosen!=11:
            new=predictions(chosen,task);old=tuple(x.copy() for x in new)
            old[0]['new']=old[0].Y_pred;old[1]['blend']=old[1].baseline;old[2]['blend']=old[2].baseline
            check=checks_for(task,old,new)
            assert all(r['coverage_sufficient'] and r['primary_pass'] and r['secondary_pass'] for r in check)
            all_checks+=check
    old=list(csv.reader(io.StringIO(OLD.read_bytes().decode('utf-8-sig'),newline='')))
    rows=[r.copy() for r in old];chosen={r['task']:r['version'] for r in routes}
    sources={v:list(csv.reader(io.StringIO(Path(verified[v]['submission']).read_bytes().decode('utf-8-sig'),newline=''))) for v in [12,13]}
    for i,row in enumerate(rows[1:],1):
        v=chosen[row[0]]
        if v!=11:
            source=sources[v][i];assert row[:4]+row[5:]==source[:4]+source[5:];row[4]=source[4]
    stream=io.StringIO(newline='');csv.writer(stream,lineterminator='\r\n').writerows(rows);blob=b'\xef\xbb\xbf'+stream.getvalue().encode()
    UPLOAD.parent.mkdir(exist_ok=True)
    if UPLOAD.exists():assert UPLOAD.read_bytes()==blob
    UPLOAD.write_bytes(blob)
    template=pd.read_csv('admet_official/submission_template.csv',dtype=str,keep_default_na=False)
    frame=pd.read_csv(UPLOAD,dtype=str,keep_default_na=False)
    assert frame.drop(columns='y_pred').equals(template.drop(columns='y_pred')) and len(frame)==16379 and frame.task.nunique()==22
    assert blob.count(b'\r\n')==16380 and blob.count(b'task,sample_idx,drug_id,smiles,y_pred,metric')==1
    for route in routes:
        task=route['task'];piece=frame[frame.task.eq(task)];q=piece.y_pred.astype(float).to_numpy()
        assert np.isfinite(q).all()
        np.testing.assert_allclose(q,pd.read_csv(ROOT/task/f'submission_{task}.csv').Y,atol=1e-12,rtol=1e-12)
        if piece.metric.iloc[0] in ['AUROC','AUPRC']:assert ((q>=0)&(q<=1)).all()
        for a,b in zip(old[1:],rows[1:]):
            if a[0]==task and not route['changed']:assert a==b
    save(ROOT/'routing.json',routes)
    result=dict(status='VERIFIED_COMBINED',submission=str(UPLOAD),sha256=sha(UPLOAD),baseline_score=49.381036,
        minimum_requested_score=49.981036,target_score=51.584929,platform_score=None,automatic_submission=False,
        rows=16379,tasks=22,columns=6,encoding='UTF-8 BOM',line_endings='CRLF',
        changed=[r['task'] for r in routes if r['changed']],unchanged=[r['task'] for r in routes if not r['changed']],
        anchor_checks=len(all_checks),overlap_checks=len(overlap_checks),primary_or_secondary_tolerance_failures=0,
        caveat='Historical validation reused; no guarantee of hidden score or +0.6 improvement; no new per-molecule routing.')
    save(ROOT/'verification.json',result);pd.DataFrame(all_checks).to_csv('admet_diagnostics/v14_anchor_checks.csv',index=False)
    if overlap_checks:pd.DataFrame(overlap_checks).to_csv('admet_diagnostics/v14_overlap_checks.csv',index=False)
    print(json.dumps(result),flush=True)
    with tarfile.open('admet_v14_review.tar.gz','w:gz') as archive:
        paths=[p for p in ROOT.rglob('*') if p.is_file()]+[UPLOAD]+list(Path('admet_diagnostics').glob('v14_*'))
        for p in paths:archive.add(p,arcname=str(p),recursive=False)


if __name__=='__main__':main()
