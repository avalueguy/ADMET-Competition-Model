"""Normalize numpy scalars at JSON boundary, preserving signed training source.

No model, target, selection, prediction, or acceptance calculation changes.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import v94_isolated_assay as v

def native(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:native(z) for k,z in x.items()}
    if isinstance(x,(list,tuple)):return [native(z) for z in x]
    return x

if __name__=='__main__':
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    original=v.b.save
    v.b.save=lambda p,x:original(p,native(x))
    v.emit=lambda **x:print(json.dumps(native(x),ensure_ascii=False),flush=True)
    v.confirm(v.load())
