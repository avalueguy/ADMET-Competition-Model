import os
from pathlib import Path
root=Path('/home/czp123/admet_v23_20260928')
assert os.name!='nt' and root.is_dir()
(root/'admet_model_outputs_v46').mkdir(exist_ok=True)
print('v46 remote directory ready')
