"""Structure-only graph message-passing experts for a fresh ADMET route.

The graph is built from the official original structures only.  Each fold fits
the neural head on the fold's training molecules; held-out labels are never
used to construct features or choose predictions.  This route is intentionally
independent of the carried fingerprint and tree models.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib, json, sys, time
from pathlib import Path
import joblib, numpy as np, torch
from torch import nn
from scipy.special import expit
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
import admet_v67_data as chemistry
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v172'
SEEDS=(1724026,1724027)
CONFIGS=(dict(hidden=96,layers=3,steps=700),)
METHOD='Independent structure-only atom-message graph experts, mean+max+sqrt-sum pooling, two seeds, fixed training duration and .2/.45 baseline blends. Revisits graph modeling with a different readout from v95; official original structures only, no assay covariates or historical predictions.'

def sig(): return hashlib.sha256((b.sha(__file__)+v.sig()+b.sha(runner.__file__)+b.sha(chemistry.__file__)).encode()).hexdigest()
def key(c): return 'h'+str(c['hidden'])+'_l'+str(c['layers'])

def atom_features(a):
    z=[1,5,6,7,8,9,15,16,17,35,53]
    out=[float(a.GetAtomicNum() in [q]) for q in z]
    out += [float(a.GetAtomicNum() not in z), float(a.GetDegree())/5., float(a.GetTotalNumHs())/4.,
            float(a.GetFormalCharge())/3., float(a.GetIsAromatic()), float(a.IsInRing())]
    hs=str(a.GetHybridization())
    out += [float(hs==q) for q in ('SP','SP2','SP3','SP3D','SP3D2')]
    return out

def bond_features(e):
    bt=e.GetBondType(); return [float(bt==q) for q in (Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC)] + [float(e.GetIsConjugated()),float(e.IsInRing())]

def build_graphs(d):
    if '_v172_graphs' in d: return d['_v172_graphs']
    cache=OUT/'graphs.joblib'; schema=OUT/'graph_schema.json'; OUT.mkdir(exist_ok=True)
    if cache.exists() and schema.exists():
        z=joblib.load(cache); assert json.loads(schema.read_text())['signature']==sig(); assert len(z)==len(d['views']['physchem']); d['_v172_graphs']=z; return z
    # mi indexes unique ORIGINAL strings, whereas canonical is a row vector.
    # Reconstruct in the same row order used by the official loader, retaining
    # stereochemistry, salts and charges rather than canonicalizing them away.
    structures=[None]*len(d['views']['physchem']); cursor=0
    for task in d['tasks']:
        for part in ('train','valid','test'):
            rows=([{'Drug':r['smiles']} for r in d['template'] if r['task']==task] if part=='test' else b.table(ROOT/'admet_competition_data/data_benchmark'/task/(part+'.csv')))
            for row in rows:
                index=int(d['mi'][cursor]); cursor+=1
                assert structures[index] in (None,row['Drug']); structures[index]=row['Drug']
    assert cursor==len(d['y']) and all(s is not None for s in structures)
    source_sha=hashlib.sha256(json.dumps(structures).encode()).hexdigest()
    expected=json.loads((ROOT/'admet_model_outputs_v67/features/official_chemistry.json').read_text())['smiles_sha256']; assert source_sha==expected
    graphs=[]; exceptions=[]
    for i,s in enumerate(structures):
        m,mode=chemistry.parse_molecule(s)
        if mode!='full_sanitization': exceptions.append(dict(index=i,mode=mode))
        xa=np.asarray([atom_features(a) for a in m.GetAtoms()],np.float32)
        src=[]; dst=[]; eb=[]
        for e in m.GetBonds():
            u,j=e.GetBeginAtomIdx(),e.GetEndAtomIdx(); ef=bond_features(e)
            src += [u,j]; dst += [j,u]; eb += [ef,ef]
        if not src: src=[0]; dst=[0]; eb=[[1.,0.,0.,0.,0.,0.]]
        graphs.append(dict(x=xa,src=np.asarray(src,np.int64),dst=np.asarray(dst,np.int64),e=np.asarray(eb,np.float32)))
    joblib.dump(graphs,cache,compress=3); b.save(schema,dict(signature=sig(),molecules=len(graphs),original_structure_sha256=source_sha,exceptions=exceptions,atom_dim=graphs[0]['x'].shape[1],bond_dim=graphs[0]['e'].shape[1])); d['_v172_graphs']=graphs; return graphs

class Net(nn.Module):
    def __init__(self, ad, ed, hidden, layers):
        super().__init__(); self.a=nn.Linear(ad,hidden); self.e=nn.Linear(ed,hidden); self.msg=nn.ModuleList([nn.Linear(hidden*2,hidden) for _ in range(layers)]); self.up=nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(layers)]); self.head=nn.Sequential(nn.Linear(hidden*3,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.12),nn.Linear(96,1))
    def forward(self,x,src,dst,edge,which):
        h=torch.relu(self.a(x)); ee=torch.relu(self.e(edge))
        for ml,ul in zip(self.msg,self.up):
            msg=torch.relu(ml(torch.cat([h[src],ee],1))); agg=torch.zeros_like(h); agg.index_add_(0,dst,msg); h=torch.relu(h+ul(agg))
        n=int(which.max().item())+1 if len(which) else 0; sm=torch.zeros((n,h.shape[1]),device=h.device); sm.index_add_(0,which,h); ct=torch.bincount(which,minlength=n).float().clamp_min(1).unsqueeze(1); mean=sm/ct
        mx=torch.full_like(mean,-1e9); mx.scatter_reduce_(0,which[:,None].expand_as(h),h,reduce='amax',include_self=True); return self.head(torch.cat([mean,mx,sm/torch.sqrt(ct)],1)).squeeze(1)

def batch(graphs, ids, device):
    xs=[]; ss=[]; dd=[]; es=[]; ww=[]; off=0
    for j,i in enumerate(ids):
        g=graphs[int(i)]; xs.append(g['x']); ss.append(g['src']+off); dd.append(g['dst']+off); es.append(g['e']); ww.append(np.full(len(g['x']),j,np.int64)); off+=len(g['x'])
    return [torch.as_tensor(np.concatenate(z),device=device) for z in (xs,ss,dd,es,ww)]

def infer(net,graphs,molecule_ids,device):
    net.eval(); raw=[]
    with torch.inference_mode():
        for lo in range(0,len(molecule_ids),96):
            raw.append(net(*batch(graphs,molecule_ids[lo:lo+96],device)).cpu().numpy())
    return np.concatenate(raw)

def loss(metric,p,y):
    if metric in ('AUROC','AUPRC'):
        z=nn.functional.binary_cross_entropy_with_logits(p,y); pos=p[y>.5]; neg=p[y<.5]
        if len(pos) and len(neg): z=z+.2*nn.functional.softplus(neg[:,None]-pos[None]).mean()
        return z
    if metric=='MAE': return (p-y).abs().mean()+.001*p.square().mean()
    delta=y[:,None]-y[None]; mask=delta.abs()>.1; z=.5*(p-y).square().mean()
    if mask.any(): z=z+.4*nn.functional.softplus(-delta.sign()*(p[:,None]-p[None]))[mask].mean()
    return z

def predict(d,k,part,c):
    folder=OUT/'models'/d['tasks'][k]/part/key(c); file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file); assert str(z['signature'])==sig(); np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]); return z['p'].copy()
    graphs=build_graphs(d); fit,q=v.split(d,k,part); device='cuda' if torch.cuda.is_available() else 'cpu'; metric=d['metrics'][k]; y=d['y'][fit]; center=float(np.median(y)) if metric=='MAE' else 0.; scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if metric=='MAE' else 1.; target=y if metric in ('AUROC','AUPRC') else (r83.ranknorm(y) if metric=='Spearman' else (y-center)/scale)
    folder.mkdir(parents=True,exist_ok=True); members=[]; start=time.monotonic()
    for seed in SEEDS:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng=np.random.default_rng(seed); net=Net(graphs[0]['x'].shape[1],graphs[0]['e'].shape[1],c['hidden'],c['layers']).to(device); opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.025); schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,c['steps'],eta_min=.00003)
        for step in range(c['steps']):
            ix=rng.choice(len(fit),size=min(96,len(fit)),replace=len(fit)<96); ids=fit[ix]; x,src,dst,e,w=batch(graphs,d['mi'][ids],device); pred=net(x,src,dst,e,w); yy=torch.as_tensor(target[ix],device=device,dtype=torch.float32); z=loss(metric,pred,yy); assert torch.isfinite(z); opt.zero_grad(set_to_none=True); z.backward(); nn.utils.clip_grad_norm_(net.parameters(),2.); opt.step(); schedule.step()
        raw=infer(net,graphs,d['mi'][q],device); mp=folder/f'model_{seed}.pt'; torch.save({'signature':sig(),'state':net.state_dict(),'hidden':c['hidden'],'layers':c['layers']},mp)
        replay=Net(graphs[0]['x'].shape[1],graphs[0]['e'].shape[1],c['hidden'],c['layers']).to(device); replay.load_state_dict(torch.load(mp,weights_only=True)['state']); np.testing.assert_array_equal(raw,infer(replay,graphs,d['mi'][q],device)); members.append(raw); del net,replay,opt
    p=np.mean(members,axis=0); p=expit(p) if metric in ('AUROC','AUPRC') else p*scale+center
    if d['tasks'][k]=='ppbr_az': p=p.clip(0,100)
    np.savez_compressed(file,signature=sig(),fit=fit,query=q,p=p,center=center,scale=scale); print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,seconds=round(time.monotonic()-start))),flush=True); return p

def main():
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
    runner.run(sys.modules[__name__],v.load())
if __name__=='__main__': main()
