"""Post-selection uncertainty and chemistry-shift diagnostics, not tuning."""
import json
import numpy as np
import admet_v24_models as b
from admet_v33_run import ROOT,OUT,P,signature,variants,aggregate,retrieve

def main():
    b.remote_only();d=b.build_data();r=json.loads((OUT/'analysis.json').read_text())
    assert r['signature']==signature() and r['status']=='EVALUATED'
    base=np.load(OUT/'baseline_valid.npy');records=[];rng=np.random.default_rng(330699)
    for record in r['tasks']:
        s=record['selection']
        if s is None:continue
        t=record['task'];k=d['tasks'].index(t);m=record['metric']
        tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1));te=np.flatnonzero((d['tid']==k)&(d['split']==2))
        z=np.load(OUT/t/'valid/predictions.npz');np.testing.assert_array_equal(z['query'],vi)
        p=variants(z['p'],m)[s['variant']];merged=b.blend(m,base[vi],p,s['weight'])
        refs,y=aggregate(d,tr);tf=b.Target(y,m)
        sim=retrieve(d,d['mi'][vi],refs,tf.encode(y),True)[:,2]
        fit=np.flatnonzero((d['tid']==k)&(d['split']<2));rr,yy=aggregate(d,fit)
        test_sim=retrieve(d,d['mi'][te],rr,b.Target(yy,m).encode(yy),True)[:,2]
        slices=[]
        for label,mask in [('distant',sim<.3),('closer',sim>=.3)]:
            if mask.sum()<8 or (m in ['AUROC','AUPRC'] and len(np.unique(d['y'][vi][mask]))<2):
                slices.append(dict(label=label,rows=int(mask.sum()),gain=None));continue
            slices.append(dict(label=label,rows=int(mask.sum()),gain=b.delta(m,d['y'][vi][mask],base[vi][mask],merged[mask])))
        groups,inv=np.unique(d['rg'][vi],return_inverse=True);blocks=[np.flatnonzero(inv==j) for j in range(len(groups))]
        draws=[]
        for _ in range(300):
            ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
            if m in ['AUROC','AUPRC'] and len(np.unique(d['y'][vi][ix]))<2:continue
            draws.append(b.delta(m,d['y'][vi][ix],base[vi][ix],merged[ix]))
        records.append(dict(task=t,metric=m,valid_gain=s['valid_gain'],valid_slice_gains=slices,
            valid_similarity_quantiles=np.quantile(sim,[.1,.5,.9]).tolist(),test_similarity_quantiles=np.quantile(test_sim,[.1,.5,.9]).tolist(),
            valid_distant_fraction=float(np.mean(sim<.3)),test_distant_fraction=float(np.mean(test_sim<.3)),
            selected_candidate_scaffold_bootstrap_95_interval=np.quantile(draws,[.025,.975]).tolist(),
            bootstrap_positive_fraction=float(np.mean(np.array(draws)>0)),
            bootstrap_post_selection=True,bootstrap_accounts_for_model_selection=False))
        b.emit('v33_diagnostic',task=t,valid_gain=s['valid_gain'],slices=slices)
    b.save(OUT/'uncertainty_and_shift.json',dict(signature=signature(),tasks=records,
        note='Post-selection bootstrap over reused historical valid; does not remove selection bias. Diagnostics never change frozen selection.'))
if __name__=='__main__':main()
