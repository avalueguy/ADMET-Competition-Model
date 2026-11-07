"""Extract transferred plain logs with traversal and duplicate-content checks."""
from pathlib import Path
import tarfile,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    records=[]
    for v in('v386','v387'):
        exp=ROOT/f'experiments/regression9_{v}'
        with tarfile.open(ROOT/f'releases/{v}_prefit_logs.tar.gz')as tf:
            seen=set()
            for member in tf.getmembers():
                if member.isdir():continue
                target=(exp/member.name).resolve()
                assert member.isfile()and target.is_relative_to((exp/'prefit_logs').resolve())and member.name not in seen
                seen.add(member.name);payload=tf.extractfile(member).read()
                if target.exists():assert target.read_bytes()==payload
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
        assert len(seen)==18
        records.append(dict(version=v,logs=len(seen)))
    print(json.dumps(dict(status='PASS',records=records)))
if __name__=='__main__':main()
