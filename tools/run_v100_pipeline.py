"""Sequential GPU kernel development, freeze, confirmation."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for stage in ('smoke','develop','confirm'):
    subprocess.run([sys.executable,'-u',str(ROOT/'tools/v100_shape_kernel.py'),stage],cwd=ROOT,check=True)
