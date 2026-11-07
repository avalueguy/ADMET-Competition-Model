"""Check ordered archive overlay can restore current inputs and selected models."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    release=json.loads((ROOT/'docs/v388_release.json').read_text(encoding='utf-8'))
    overlay={};sources=[]
    for receipt in release['dependencies']['required_archives']+[release]:
        archive=ROOT/receipt['archive'];assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
        with tarfile.open(archive)as tf:
            for m in tf.getmembers():
                if m.isfile():overlay[m.name.removeprefix('./')]=(archive,m.name)
        sources.append(receipt['archive'])
    pool_hashes={}
    for pool in('v377_regression9_delta.tar.gz','v371_expert_inputs.tar.gz','v374_delta_inputs.tar.gz','v375_delta_inputs.tar.gz'):
        with tarfile.open(ROOT/'releases'/pool)as tf:
            for member in tf:
                name=member.name.removeprefix('./')
                if(member.isfile()or member.islnk())and name.endswith('.npz'):
                    digest=hashlib.sha256(tf.extractfile(member).read()).hexdigest()
                    pool_hashes[(name,digest)]='releases/'+pool+'::'+name
    checked=[]
    for v,previous in(('v386','v385'),('v387','v386')):
        exp=ROOT/f'experiments/regression9_{v}';manifest=json.loads((exp/'data_manifest.json').read_text(encoding='utf-8'))
        pairs=[('official_rows.npz',manifest['official_rows_sha256']),('expert_features.npz',manifest['expert_features_sha256']),
            ('stereo_keys.npz',manifest['stereo_keys_sha256']),('v385_baseline.csv',manifest['baseline_csv_sha256'])]
        pairs += [(r[k],r[k+'_sha256'])for r in manifest['controls']for k in('control','design')]
        for name,expected in pairs:
            full=f'experiments/regression9_{v}/{name}';origin=full
            for sourcev in(v,previous,'v385','v384','v381','v380','v377'):
                option=f'experiments/regression9_{sourcev}/{name}'
                if option in overlay:origin=option;break
            if origin in overlay:
                archive,member=overlay[origin]
                with tarfile.open(archive)as tf:digest=hashlib.sha256(tf.extractfile(member).read()).hexdigest()
            else:
                assert(name,expected)in pool_hashes,(v,name,'absent in nested dependency inputs')
                digest=expected;origin=pool_hashes[(name,expected)]
            assert digest==expected,(v,name,origin)
            checked.append(dict(destination=full,source_member=origin,sha256=digest))
        final=[p for p in(exp/'outputs').rglob('state.joblib')if'final'in p.parts]
        assert final
        for p in final:
            key=p.relative_to(ROOT).as_posix();assert key in overlay
            archive,member=overlay[key]
            with tarfile.open(archive)as tf:assert hashlib.sha256(tf.extractfile(member).read()).hexdigest()==hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(status='PASS',archive_order=sources,input_hashes_checked=len(checked),selected_final_states_included=True,records=checked,no_fitting_performed=True)
    (ROOT/'docs/v388_dependency_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items()if k!='records'}))
if __name__=='__main__':main()
