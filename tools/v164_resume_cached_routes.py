"""Wait for disjoint development shards before the original frozen runner."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPECS={
 '153':dict(folder='admet_model_outputs_v153',key='depth5',script='v153_path_latent_trees.py'),
 '159':dict(folder='admet_model_outputs_v159',key='aux0.3',script='v159_multitask_transfer.py')}
def main():
    spec=SPECS[sys.argv[1]];last=None
    for _ in range(720):
        paths=list((ROOT/spec['folder']/'models').glob('*/fold1/'+spec['key']+'/predictions.npz'));count=len(paths)
        if count!=last:print(json.dumps(dict(route=sys.argv[1],cached_development_tasks=count)),flush=True);last=count
        if count==21:break
        time.sleep(20)
    else:raise RuntimeError('Shard completion timeout; preserve artifacts.')
    subprocess.run([sys.executable,str(ROOT/'tools'/spec['script'])],cwd=ROOT,check=True)
if __name__=='__main__':main()
