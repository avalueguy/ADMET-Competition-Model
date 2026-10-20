"""Label-free CPU feature preparation while the isolated neural GPU job continues."""
import os
import admet_v24_models as b
import admet_v41_run as v
assert os.name != 'nt' and str(v.ROOT) == '/home/czp123/admet_v23_20260928'
d = b.build_data()
v.chemistry(d)
print('Label-free parent molecular features cached', flush=True)
