"""Collect lightweight evidence only, never GPU checkpoints."""
import tarfile
from pathlib import Path
from admet_v32_run import ROOT,OUT

dest=ROOT/'admet_submissions/v32'
if dest.exists():
    files=list(dest.glob('*.csv'))
    assert len(files)==1
    name='我从山北来带着兰花草-小分子药物预测技术创新赛-20261006.csv'
    target=dest/name
    if files[0]!=target:
        assert not target.exists()
        files[0].rename(target)
files=list(OUT.glob('*.json'))+list(OUT.glob('*.log'))+list(OUT.glob('*/valid/metadata.json'))+list(OUT.glob('*/final/metadata.json'))+list(OUT.glob('*/fold*/metadata.json'))
files+=list(dest.glob('*'))
with tarfile.open(ROOT/'admet_v32_evidence.tar.gz','w:gz') as tar:
    for file in files:
        if file.is_file():tar.add(file,arcname=file.relative_to(ROOT).as_posix(),recursive=False)
print(len(files),(ROOT/'admet_v32_evidence.tar.gz').stat().st_size)
