"""Verify frozen source provenance before combining completed experiments."""
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rows=[]
    exceptions=read('admet_v12_resource_revision.json')['files']
    for v in [12, 13]:
        for path, original_hash in read(f'admet_v{v}_source_freeze.json').items():
            expected=exceptions.get(path, original_hash) if v==12 else original_hash
            actual=sha(path)
            assert actual==expected, path
            rows.append(dict(path=path, expected=expected, actual=actual,
                runtime_revision=expected!=original_hash))
    combination=read('admet_v14_combination_protocol.json')
    assert sha('admet_v14_combine.py')==combination['code_sha256']
    for v in [12, 13]:
        out=Path(f'admet_model_outputs_v{v}')
        assert read(out/'finish_status.json')['status']=='COMPLETE_VERIFIED'
        assert len(list(out.glob('*/decision.json')))==22
        assert len(list(out.glob('*/*_cv.json')))==66
        launch=read(out/'launch.json')
        for name in ['pid', 'finisher_pid']:
            proc=Path('/proc')/str(launch[name])
            if proc.exists() and proc.joinpath('stat').read_text().split()[2]!='Z':
                assert f'admet_v{v}_'.encode() not in proc.joinpath('cmdline').read_bytes(), (v,name)
        verification=read(f'admet_release_v{v}/independent_verification.json')
        assert verification['status']=='VERIFIED' and verification['independent_replay']
        assert sha(verification['submission'])==verification['sha256']
        audit=read(f'admet_diagnostics/v{v}_anchor_audit.json')
        assert audit['status']=='PASSED' and not audit['violations'] and not audit['insufficient_coverage']
    result=dict(status='PASSED', source_hashes=rows,
        combination_protocol_sha256=sha('admet_v14_combination_protocol.json'),
        combination_code_sha256=sha('admet_v14_combine.py'),
        note='The only frozen-source exceptions are documented GPU memory and worker scheduling changes.')
    Path('admet_diagnostics/v14_source_preflight.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'], source_files=len(rows))))


if __name__=='__main__':main()
