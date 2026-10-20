"""No training: independently reload selected final source models on the remote GPU."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import json
from pathlib import Path
import numpy as np
import torch,joblib
import admet_v24_models as b
import admet_v58_train as seq
import admet_v59_kernel as kernel

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v60'

def main():
    b.remote_only();d=b.build_data();replay=json.loads((OUT/'fusion_replay.json').read_text());checks=[];views=None
    for task,r in replay.items():
        k=d['tasks'].index(task);metric=d['metrics'][k];rr=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2))
        if r['family']=='kernel':
            dest=kernel.OUT/task/'final';z=np.load(dest/'predictions.npz');matrices,_=kernel.matrices(d,rr,q,'final');parts=[]
            for config,(family,c) in enumerate(kernel.CONFIGS):
                saved=joblib.load(dest/f'config{config}.joblib');assert saved['signature']==kernel.signature()
                np.testing.assert_array_equal(saved['rows'],rr);model=saved['model'];xx=matrices[family][1];cls=metric in ['AUROC','AUPRC']
                p=model.decision_function(xx) if cls else model.predict(xx)
                single=model.decision_function(xx[:1]) if cls else model.predict(xx[:1])
                np.testing.assert_allclose(p[:1],single,atol=1e-9,rtol=1e-9)
                if metric=='MAE':p=saved['target'].decode(p)
                parts.append(p)
            raw=np.stack(parts);np.testing.assert_allclose(raw,z['p'],atol=1e-9,rtol=1e-9);branch=raw[r['selection']['variant']]
        else:
            if views is None:views=seq.tokens(d)
            dest=seq.OUT/task/'final';z=np.load(dest/'predictions.npz');mols=np.unique(d['mi'][np.r_[rr,q]]);lookup=np.full(len(d['smiles']),-1);lookup[mols]=np.arange(len(mols));qq=lookup[d['mi'][q]]
            fp=np.concatenate([d['fp2'][mols],d['fcfp'][mols]],1);rawx=np.concatenate([d['desc'][mols],d['emb'][mols]],1);parts=[]
            for seed in seq.P['seeds']:
                snaps=[]
                for step in seq.P['snapshots']:
                    ck=torch.load(dest/f'{seed}_{step}.pt',map_location='cpu',weights_only=False);assert ck['signature']==seq.signature()
                    net=seq.Net(*ck['dimensions']).cuda().eval();net.load_state_dict(ck['state'])
                    xs=[torch.tensor(x,device='cuda') for x in [fp,ck['pre'].transform(rawx)]];values=[]
                    with torch.inference_mode():
                        for s in range(0,len(q),48):
                            with torch.autocast('cuda',dtype=torch.bfloat16):p=net(seq.batch(views,d['mi'][q[s:s+48]],0),*[x[qq[s:s+48]] for x in xs])
                            values.append(p.float().cpu().numpy())
                    p=np.concatenate(values,axis=1)
                    if metric=='MAE':p=ck['target'].decode(p)
                    snaps.append(p);del net,xs
                parts.append(np.mean(snaps,0))
            raw=np.mean(parts,0);np.testing.assert_allclose(raw,z['p'],atol=2e-5,rtol=2e-5);branch=seq.variants(raw,metric)[r['selection']['variant']]
        np.testing.assert_array_equal(z['fit'],rr);np.testing.assert_array_equal(z['query'],q)
        np.testing.assert_allclose(branch,r['branch'],atol=2e-5,rtol=2e-5)
        result=b.blend(metric,np.array(r['baseline']),branch,r['selection']['weight']);np.testing.assert_allclose(result,r['result'],atol=2e-6,rtol=2e-6)
        checks.append(dict(task=task,family=r['family'],model_reload_passed=True,fit_query_indices_exact=True,test_labels_used=False))
        b.emit('v60_model_replay',**checks[-1]);torch.cuda.empty_cache()
    b.save(OUT/'model_replay.json',dict(status='PASS',tasks=checks,local_gpu_training=False,test_labels_used=False))

if __name__=='__main__':main()
