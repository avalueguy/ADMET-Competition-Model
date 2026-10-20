"""Fresh process replays final GINE snapshots and exported task predictions."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import json
import numpy as np,torch
import admet_v24_models as b
import admet_v27_models as graph
import admet_v62_model as v
from admet_v62_run import variants

def main():
    b.remote_only();torch.backends.cuda.matmul.allow_tf32=False;d=b.build_data();g=graph.graphs(d);rr=np.flatnonzero(d['split']<2);q=np.flatnonzero(d['split']==2);parts=[]
    for seed in v.P['seeds']:
        snaps=[];dest=v.OUT/'final'/str(seed);z=np.load(dest/'predictions.npz');np.testing.assert_array_equal(z['fit'],rr);np.testing.assert_array_equal(z['query'],q)
        for step in v.P['snapshot_steps']:
            ck=torch.load(dest/f'checkpoint_{step}.pt',map_location='cpu',weights_only=False);assert ck['signature']==v.signature();net=v.Net(*ck['dimensions']).cuda().eval();net.load_state_dict(ck['state']);raw=np.concatenate([d['desc'],d['emb']],1);xx=torch.tensor(np.concatenate([d['fp2'],d['fcfp'],ck['pre'].transform(raw)],1),device='cuda');snaps.append(v.predict(net,d,g,xx,q,ck['target']));del net,xx;torch.cuda.empty_cache()
        part=np.mean(snaps,0);np.testing.assert_allclose(part,z['p'],atol=3e-5,rtol=3e-5);parts.append(part)
    pp=np.mean(parts,0);replay=json.loads((v.OUT/'fusion_replay.json').read_text());checks=[]
    for task,r in replay.items():
        k=d['tasks'].index(task);m=d['metrics'][k];s=r['selection'];active=d['tid'][q]==k;branch=variants(pp[:,active],m)[s['variant']];np.testing.assert_allclose(branch,r['branch'],atol=3e-5,rtol=3e-5);merged=b.blend(m,np.array(r['baseline']),branch,s['weight'])
        if task=='ppbr_az':merged=merged.clip(0,100)
        np.testing.assert_allclose(merged,r['result'],atol=3e-5,rtol=3e-5);checks.append(dict(task=task,model_reload_passed=True,fit_query_indices_exact=True));b.emit('v62_model_replay',**checks[-1])
    b.save(v.OUT/'model_replay.json',dict(status='PASS',tasks=checks,test_labels_used=False,local_gpu_training=False))

if __name__=='__main__':main()
