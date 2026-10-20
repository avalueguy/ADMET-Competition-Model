"""Replay final learned sources independently, with feature-label isolation probes."""
import json
from pathlib import Path
import numpy as np
import torch
import admet_v24_models as b
import admet_v27_models as graph
import admet_v46_run as adapter
import admet_v51_run as trees
import admet_v52_neural as neural
import admet_v54_graph as adapt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v56'


def main():
    b.remote_only();d=b.build_data();c=trees.chemistry(d);g=None;records=[]
    replay=json.loads((OUT/'fusion_replay.json').read_text())
    for task,s in replay.items():
        family=s['family']
        if family is None:continue
        k=d['tasks'].index(task);rr=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2))
        if family=='neural':
            dest=neural.OUT/task/'final';z=np.load(dest/'predictions.npz')
            unique=np.unique(d['mi'][np.r_[rr,q]]);lookup=np.full(len(d['smiles']),-1);lookup[unique]=np.arange(len(unique));qq=lookup[d['mi'][q]]
            raw=np.concatenate([d['desc'],d['emb'],c['prior']],1)
            fp=np.concatenate([d['fp2'][unique],d['fcfp'][unique]],1)
            prior=c['prior'][unique,0] if task=='lipophilicity_astrazeneca' else (c['prior'][unique,1] if task=='solubility_aqsoldb' else np.zeros(len(unique)))
            seeds=[]
            for seed in neural.SEEDS:
                families=[]
                for f in range(2):
                    snaps=[]
                    for epoch in [90,115,140]:
                        ck=torch.load(dest/f'family{f}_{seed}_{epoch}.pt',map_location='cpu',weights_only=False)
                        dense=ck['pre'].transform(raw[unique]);xs=[torch.tensor(x,dtype=torch.float32,device='cuda') for x in [fp,dense,prior]]
                        net=neural.Net(*ck['dimensions'],ck['family'],task).cuda().eval();net.load_state_dict(ck['state'])
                        with torch.inference_mode():
                            p=np.concatenate([net(*[x[qq[a:a+512]] for x in xs],ck['center'],ck['scale']).cpu().numpy() for a in range(0,len(q),512)])
                            single=net(*[x[qq[:1]] for x in xs],ck['center'],ck['scale']).cpu().numpy()
                        np.testing.assert_allclose(p[:1],single,atol=2e-5,rtol=2e-5)
                        snaps.append(p);del net,xs
                    families.append(np.mean(snaps,0))
                seeds.append(np.stack(families))
            pred=np.mean(seeds,0)
            np.testing.assert_allclose(pred,z['p'],atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(pred,s['raw'],atol=2e-5,rtol=2e-5)
        else:
            if g is None:g=graph.graphs(d)
            dest=adapt.OUT/task/'final';z=np.load(dest/'predictions.npz');seeds=[]
            for seed in adapt.P['seeds']:
                snaps=[]
                for step in adapt.P['snapshots']:
                    ck=torch.load(dest/f'{seed}_{step}.pt',map_location='cpu',weights_only=False)
                    net=graph.BondNet(*ck['dimensions']).cuda().eval();net.load_state_dict(ck['state'])
                    raw=np.concatenate([d['desc'],d['emb']],1)
                    x=torch.tensor(np.concatenate([d['fp2'],d['fcfp'],ck['preprocessor'].transform(raw)],1),device='cuda')
                    p=adapter.logits(net,g,x,d['mi'][q])[:,:,k]
                    if d['metrics'][k]=='MAE':p=ck['target'].decode(p)
                    snaps.append(p);del net,x
                seeds.append(np.mean(snaps,0))
            pred=np.mean(seeds,0);np.testing.assert_allclose(pred,z['p'],atol=3e-5,rtol=3e-5)
            np.testing.assert_allclose(pred,s['raw'],atol=3e-5,rtol=3e-5)
        np.testing.assert_array_equal(z['fit'],rr);np.testing.assert_array_equal(z['query'],q)
        records.append(dict(task=task,family=family,final_model_reload_passed=True,fit_query_indices_match=True,
            predictions_match_saved_source=True,test_labels_used=False))
        b.emit('v56_verify',task=task,family=family);torch.cuda.empty_cache()
    b.save(OUT/'model_replay.json',dict(status='PASS',tasks=records,local_gpu_training=False,test_labels_used=False))


if __name__=='__main__':main()
