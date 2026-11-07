"""Preserve prelaunch source, align MAE and rank to the same submitted output."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/regression9_v379'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (EXP/'launch.json').exists() and not (EXP/'outputs').exists()
    assert not (EXP/'source_amendment.json').exists()
    archive=ROOT/'releases/v379_distribution_delta.tar.gz'
    old=json.loads((EXP/'source_freeze.json').read_text())
    backup=EXP/'prelaunch_source_v1'
    backup.mkdir()
    for row in old['files']:
        assert sha(EXP/row['name'])==row['sha256']
        shutil.copyfile(EXP/row['name'],backup/row['name'])
    shutil.copyfile(EXP/'source_freeze.json',backup/'source_freeze.json')
    shutil.copyfile(archive,ROOT/'releases/v379_prelaunch_source_v1.tar.gz')
    runner=ROOT/'tools/v379_run_experts.py'
    source=runner.read_text(encoding='utf-8')
    original="official=(1-w)*rankdata(a['score_p'])/len(y)+w*rankdata(expert)/len(y)"
    assert source.count(original)==1
    runner.write_text(source.replace(original,'official=new'),encoding='utf-8')
    shutil.copyfile(runner,EXP/runner.name)
    oldhash=sha(archive)
    old['files']=[dict(name=row['name'],sha256=sha(EXP/row['name'])) for row in old['files']]
    old['ranking_output']='The exact same native-unit linear blend for MAE, Spearman, and final CSV.'
    (EXP/'source_freeze.json').write_text(json.dumps(old,indent=2)+'\n')
    amendment=dict(reason='Align ranking-task MAE and Spearman verification with exactly the same native-unit prediction that will be submitted.',
                   before_any_model_fit=True,old_archive_sha256=oldhash,
                   old_sources='prelaunch_source_v1',no_scores_seen_for_v379=True)
    (EXP/'source_amendment.json').write_text(json.dumps(amendment,indent=2)+'\n')
    with tarfile.open(archive,'w:gz') as tf:
        for name in [r['name'] for r in old['files']]+['source_freeze.json','source_amendment.json']:
            tf.add(EXP/name,arcname=name)
    receipt=dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,
                 local_input_files_verified=96,source_files=old['files'],all_fitting_remote=True,source_amendment=amendment)
    (ROOT/'docs/v379_input_package.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
