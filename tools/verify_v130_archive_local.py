"""Verify v130 archive, preserving the locally enriched v124 check report."""
import hashlib,json,tarfile
from pathlib import Path,PurePosixPath
from restore_iteration_archive import ROOT,load,sha


def main():
    prefix='v130_low_task_experts_20261027'
    archive=ROOT/'releases'/(prefix+'.tar.gz')
    mp=ROOT/'releases'/(prefix+'.manifest.json')
    info=load(ROOT/'releases'/(prefix+'.archive.json'));manifest=load(mp)
    assert sha(archive)==info['sha256'] and archive.stat().st_size==info['bytes']
    assert sha(mp)==info['manifest_sha256']
    for dep in manifest['dependencies']:assert sha(ROOT/dep['path'])==dep['sha256']
    expected={r['path']:r for r in manifest['files']};seen=set();exact=0;exceptions=[]
    anchor=next((ROOT/'admet_submissions/v124_matched_models_history').glob('*.csv'))
    old_check=anchor.with_suffix('.verification.json').relative_to(ROOT).as_posix()
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            path=PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts and member.name not in seen
            seen.add(member.name);data=tf.extractfile(member).read()
            if member.name==mp.relative_to(ROOT).as_posix():assert data==mp.read_bytes();continue
            row=expected[member.name]
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
            local=ROOT/member.name;assert local.is_file()
            if sha(local)==row['sha256']:exact+=1;continue
            # Only known format-report enrichment may differ; model/CSV bytes may not.
            assert member.name==old_check,member.name
            archived=json.loads(data);current=load(local)
            difference={k for k in archived.keys()|current.keys() if archived.get(k)!=current.get(k)}
            assert difference=={'independent_final_fusion_replay','changed_tasks','unchanged_tasks','model_validation_replay'}
            assert current['independent_final_fusion_replay'] is True
            assert archived['changed_tasks']=={'ames':1457,'half_life_obach':134}
            assert archived['unchanged_tasks']==20 and archived['model_validation_replay']=='PASS'
            assert archived['sha256']==current['sha256']==sha(anchor)
            assert current['status']=='PASS' and current['rows']==16379 and current['tasks']==22
            exceptions.append(dict(path=member.name,reason='Same immutable v124 CSV; local final-fusion check report adds a replay flag and omits earlier summary fields.',differing_keys=sorted(difference)))
    assert seen==set(expected)|{mp.relative_to(ROOT).as_posix()}
    result=dict(status='PASS',prefix=prefix,archive_sha256=sha(archive),manifest_files=len(expected),all_member_hashes_verified=True,
        dependency_hashes_verified=True,existing_exact=exact,existing_verification_metadata_preserved=exceptions)
    out=ROOT/'docs/v130_local_archive_verification.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
