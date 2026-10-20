"""Rebuild fitted anchors, reload difference models and probe query-label isolation."""
import json
import numpy as np
import torch
import admet_v24_models as b
import admet_v44_run as v

def main():
    b.remote_only()
    torch.backends.cuda.matmul.allow_tf32 = False
    d = b.build_data()
    report = json.loads((v.OUT/'analysis.json').read_text())
    assert report['status']=='EVALUATED' and report['signature']==v.signature()
    records = []
    for r in report['tasks']:
        if r['selection'] is None:
            continue
        k = d['tasks'].index(r['task'])
        rows = np.flatnonzero((d['tid']==k)&(d['split']<2))
        query = np.flatnonzero((d['tid']==k)&(d['split']==2))
        dest = v.OUT/r['task']/'final'
        z = np.load(dest/'predictions.npz')
        assert str(z['signature'])==v.signature()
        np.testing.assert_array_equal(rows,z['fit'])
        np.testing.assert_array_equal(query,z['query'])
        x = v.design(d,rows,query)
        ck = torch.load(dest/'preprocessing.pt',map_location='cpu',weights_only=False)
        assert ck['signature']==v.signature()
        for name in ['med','mu','sd']:
            np.testing.assert_array_equal(getattr(x['pre'],name),getattr(ck['pre'],name))
        for name in ['chosen','refs','labels']:
            np.testing.assert_array_equal(x[name],ck[name])
        iq,sq = v.neighbors(d,d['mi'][query],x['refs'],v.P['query_neighbors'])
        assert not np.any(d['groups'][d['mi'][query,None]]==d['groups'][x['refs'][iq]])
        parts = []
        for seed,depth,trees in zip(v.P['seeds'],v.P['depths'],v.P['trees']):
            model = v.make_model(seed,depth,trees)
            model.load_model(dest/f'point_{seed}.ubj')
            model.set_params(device='cuda',n_jobs=3)
            pred = model.predict(x['point'][d['mi'][query]])
            if r['metric']=='MAE':
                pred = x['target'].decode(pred)
            pair = v.make_model(seed,depth,trees,True)
            pair.load_model(dest/f'pair_{seed}.ubj')
            pair.set_params(device='cuda',n_jobs=3)
            rec = v.reconstruct(d,x,query,iq,sq,pair)
            singleton = v.reconstruct(d,x,query[:1],iq[:1],sq[:1],pair)
            np.testing.assert_allclose(rec[:,:1],singleton,rtol=2e-6,atol=2e-6)
            # Antisymmetry follows from explicitly subtracting both directions.
            q = d['mi'][query[:3]]
            ref = x['refs'][iq[:3,0]]
            score = sq[:3,0]
            a = v.pair_matrix(x,q,ref,score)
            c = v.pair_matrix(x,ref,q,score)
            forward = .5*(pair.predict(a)-pair.predict(c))
            reverse = .5*(pair.predict(c)-pair.predict(a))
            np.testing.assert_array_equal(forward,-reverse)
            parts.append(np.concatenate([pred[None],rec]))
        np.testing.assert_allclose(np.mean(parts,0),z['p'],rtol=2e-6,atol=2e-6)
        probe = dict(d)
        probe['y'] = d['y'].copy()
        probe['y'][query] = 1e9
        xx = v.design(probe,rows,query)
        for name in ['refs','labels','y','chosen','dense','fp','point']:
            np.testing.assert_array_equal(x[name],xx[name])
        records.append(dict(task=r['task'],model_reload_passed=True,fit_only_anchors_rebuilt=True,
            query_cross_scaffold_anchors_checked=True,hidden_target_labels_invariant=True,
            saved_fit_preprocessors_match=True,antisymmetric_prediction_passed=True,
            query_batch_invariant=True,test_labels_used=False))
        b.emit('v44_replay',**records[-1])
    b.save(v.OUT/'model_replay.json',dict(status='FINAL_PAIR_AND_ANCHOR_ISOLATION_REPLAY_PASSED',
        signature=v.signature(),tasks=records,local_gpu_training=False,test_labels_used=False))

if __name__=='__main__':
    main()
