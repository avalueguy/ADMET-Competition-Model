"""Rehydrate unchanged input files after extracting all documented dependencies."""
from pathlib import Path
import hashlib,json,os,shutil,tarfile

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    restored=[]
    dependencies=json.loads((ROOT/'docs/v388_handoff_dependencies.json').read_text(encoding='utf-8'))
    for r in dependencies['required_archives']:assert sha(ROOT/r['archive'])==r['sha256']
    chain=[('v380_complete_handoff.tar.gz','v379_complete_handoff.tar.gz'),
        ('v379_complete_handoff.tar.gz','v377_v378_complete_handoff.tar.gz'),
        ('v377_v378_complete_handoff.tar.gz','v374_v375_complete_handoff.tar.gz'),
        ('v377_v378_complete_handoff.tar.gz','v377_regression9_delta.tar.gz'),
        ('v374_v375_complete_handoff.tar.gz','v371_expert_inputs.tar.gz'),
        ('v374_v375_complete_handoff.tar.gz','v374_delta_inputs.tar.gz'),
        ('v374_v375_complete_handoff.tar.gz','v375_delta_inputs.tar.gz')]
    for parent,name in chain:
        dest=ROOT/'releases'/name
        with tarfile.open(ROOT/'releases'/parent)as tf:
            member=tf.getmember('releases/'+name);assert member.isfile()
            if dest.exists():
                h=hashlib.sha256()
                stream=tf.extractfile(member)
                while chunk:=stream.read(8*1024*1024):h.update(chunk)
                assert sha(dest)==h.hexdigest()
            else:
                with dest.open('wb')as out:shutil.copyfileobj(tf.extractfile(member),out)
                restored.append(dest.relative_to(ROOT).as_posix())
    pools=[ROOT/'releases'/n for n in('v377_regression9_delta.tar.gz','v371_expert_inputs.tar.gz','v374_delta_inputs.tar.gz','v375_delta_inputs.tar.gz')]
    for v,previous in(('v386','v385'),('v387','v386')):
        exp=ROOT/f'experiments/regression9_{v}';prev=ROOT/f'experiments/regression9_{previous}'
        manifest=json.loads((exp/'data_manifest.json').read_text(encoding='utf-8'))
        pairs=[('official_rows.npz',manifest['official_rows_sha256']),('expert_features.npz',manifest['expert_features_sha256']),
            ('stereo_keys.npz',manifest['stereo_keys_sha256']),('v385_baseline.csv',manifest['baseline_csv_sha256'])]
        pairs += [(r[k],r[k+'_sha256'])for r in manifest['controls']for k in('control','design')]
        for n,expected in pairs:
            dest=exp/n
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True)
                src=prev/n
                if src.is_file()and sha(src)==expected:
                    try:os.link(src,dest)
                    except OSError:shutil.copyfile(src,dest)
                else:
                    found=False
                    for pool in pools:
                        with tarfile.open(pool)as tf:
                            members={m.name.removeprefix('./'):m for m in tf.getmembers()if m.isfile()or m.islnk()}
                            if n not in members:continue
                            payload=tf.extractfile(members[n]).read()
                            if hashlib.sha256(payload).hexdigest()!=expected:continue
                            dest.write_bytes(payload);found=True;break
                    assert found,(v,n,'missing matching input in dependency archives')
                restored.append(dest.relative_to(ROOT).as_posix())
            assert sha(dest)==expected,(v,n)
        for r in json.loads((exp/'source_freeze.json').read_text(encoding='utf-8'))['files']:
            assert sha(exp/r['name'])==r['sha256']==sha(ROOT/'tools'/r['name'])
        template=ROOT/'admet_official/submission_template.csv'
        if not template.exists():
            with tarfile.open(ROOT/'releases/v371_expert_inputs.tar.gz')as tf:payload=tf.extractfile('submission_template.csv').read()
            template.parent.mkdir(parents=True,exist_ok=True);template.write_bytes(payload)
        local_template=exp/'submission_template.csv'
        if not local_template.exists():shutil.copyfile(template,local_template)
    print(json.dumps(dict(status='PASS',restored_files=restored,no_fitting=True,source_and_input_hashes_verified=True)))

if __name__=='__main__':main()
