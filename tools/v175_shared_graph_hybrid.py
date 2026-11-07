"""Joint graph/fingerprint experts with globally excluded confirmation scaffolds.

One frozen shared network per split learns all endpoints. No endpoint's query
identity or scaffold may occur among ANY training labels. Inputs contain only
official molecular structure. Saved models independently replay every query.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
import v172_graph_message_passing as graph
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v175'
SEEDS=(1754026,1754027)
CONFIGS=(dict(hidden=96,layers=3,steps=5000),)
METHOD='Joint all-endpoint hybrid graph/fingerprint network: bond messages, learned atom attention, mean+max readouts, six fingerprint projections, physchem descriptors, endpoint-specific heads; task-balanced optimization. Every query scaffold excluded globally across all tasks. Official structures only. Fixed .2/.45 blends and unchanged confirmation gates.'

def sig():return hashlib.sha256((b.sha(__file__)+v.sig()+b.sha(runner.__file__)+graph.sig()).encode()).hexdigest()
def key(c):return 'h'+str(c['hidden'])+'_l'+str(c['layers'])+'_s'+str(c['steps'])

def design(d,part):
    queries={k:v.split(d,k,part)[1] for k in range(len(d['tasks']))}
    allq=np.unique(np.concatenate(list(queries.values())))
    source=np.flatnonzero(d['split']<(2 if part=='final' else 1))
    if part!='final':
        source=source[~np.isin(d['groups'][source],d['groups'][allq]) & ~np.isin(d['canonical'][source],d['canonical'][allq])]
        assert not set(d['groups'][source]) & set(d['groups'][allq])
    assert not set(source)&set(allq) and np.all(np.isfinite(d['y'][source]))
    unique=np.unique(d['mi'][source]);raw=d['views']['physchem'].astype(float)
    median=np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0. for z in raw[unique].T])
    raw=np.where(np.isfinite(raw),raw,median);center=raw[unique].mean(0);scale=raw[unique].std(0).clip(.05)
    desc=np.clip((raw-center)/scale,-6,6).astype(np.float32)
    fp=np.sqrt(np.maximum(d['views']['multiview'][:,:12288],0)).astype(np.float32)
    members={k:source[d['tid'][source]==k] for k in range(len(d['tasks']))}
    targets=np.zeros(len(d['y']),np.float32);states={}
    for k,rows in members.items():
        assert len(rows)>1
        m=d['metrics'][k];y=d['y'][rows]
        yc=float(np.median(y)) if m=='MAE' else 0.;ys=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if m=='MAE' else 1.
        targets[rows]=(y-yc)/ys if m=='MAE' else r83.ranknorm(y) if m=='Spearman' else y
        states[k]=dict(center=yc,scale=ys)
    return dict(source=source,queries=queries,unique=unique,members=members,target=targets,states=states,median=median,center=center,scale=scale,desc=desc,fp=fp)

class Net(nn.Module):
    def __init__(self,ad,ed,dd,tasks,c):
        super().__init__();h=c['hidden'];self.c=c
        self.atom=nn.Linear(ad,h);self.bond=nn.Linear(ed,h)
        self.messages=nn.ModuleList(nn.Linear(2*h,h) for _ in range(c['layers']))
        self.updates=nn.ModuleList(nn.Sequential(nn.Linear(h,h),nn.LayerNorm(h)) for _ in range(c['layers']))
        self.attention=nn.Linear(h,1)
        self.fp=nn.ModuleList(nn.Sequential(nn.Linear(2048,32),nn.LayerNorm(32),nn.SiLU(),nn.Dropout(.2)) for _ in range(6))
        self.desc=nn.Sequential(nn.Linear(dd,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.15))
        self.shared=nn.Sequential(nn.Linear(3*h+256,192),nn.LayerNorm(192),nn.SiLU(),nn.Dropout(.2))
        self.heads=nn.ModuleList(nn.Sequential(nn.Linear(192,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,1)) for _ in range(tasks))
    def forward(self,x,src,dst,edge,which,fp,desc,k):
        h=torch.nn.functional.silu(self.atom(x));e=torch.nn.functional.silu(self.bond(edge))
        for ml,ul in zip(self.messages,self.updates):
            msg=torch.nn.functional.silu(ml(torch.cat([h[src],e],1)));agg=torch.zeros_like(h);agg.index_add_(0,dst,msg)
            degree=torch.bincount(dst,minlength=len(h)).float().clamp_min(1).sqrt().unsqueeze(1)
            h=torch.nn.functional.silu(h+ul(agg/degree))
        n=len(fp);sm=torch.zeros((n,h.shape[1]),device=h.device);sm.index_add_(0,which,h)
        count=torch.bincount(which,minlength=n).float().clamp_min(1).unsqueeze(1)
        mx=torch.full_like(sm,-1e9);mx.scatter_reduce_(0,which[:,None].expand_as(h),h,reduce='amax',include_self=True)
        aw=self.attention(h).clamp(-8,8).exp();aw_sum=torch.zeros((n,1),device=h.device);aw_sum.index_add_(0,which,aw)
        attended=torch.zeros_like(sm);attended.index_add_(0,which,h*aw/aw_sum[which].clamp_min(1e-8))
        molecular=torch.cat([sm/count,mx,attended]+[layer(fp[:,j*2048:(j+1)*2048]) for j,layer in enumerate(self.fp)]+[self.desc(desc)],1)
        return self.heads[k](self.shared(molecular)).squeeze(1)

def loss(m,p,y):
    if m in ('AUROC','AUPRC'):
        z=nn.functional.binary_cross_entropy_with_logits(p,y);pos=p[y>.5];neg=p[y<.5]
        if len(pos) and len(neg):z=z+.15*nn.functional.softplus(neg[:,None]-pos[None]).mean()
    elif m=='MAE':z=(p-y).abs().mean()
    else:
        delta=y[:,None]-y[None];mask=delta.abs()>.1;z=.5*(p-y).square().mean()
        if mask.any():z=z+.3*nn.functional.softplus(-delta.sign()*(p[:,None]-p[None]))[mask].mean()
    return z+.001*p.square().mean()

def infer(net,d,a,graphs,fp,desc,k,device):
    net.eval();out=[];q=a['queries'][k]
    with torch.inference_mode():
        for lo in range(0,len(q),96):
            ids=d['mi'][q[lo:lo+96]];out.append(net(*graph.batch(graphs,ids,device),fp[ids],desc[ids],k).cpu().numpy())
    return np.concatenate(out).astype(float)

def predict(d,k,part,c):
    folder=OUT/'models'/part/key(c);path=folder/'predictions.npz'
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig();np.testing.assert_array_equal(z[f'query{k}'],v.split(d,k,part)[1]);return z[f'p{k}'].copy()
    start=time.monotonic();a=design(d,part);graphs=graph.build_graphs(d);folder.mkdir(parents=True,exist_ok=True)
    device='cuda';fp=torch.as_tensor(a['fp'],device=device);desc=torch.as_tensor(a['desc'],device=device);target=torch.as_tensor(a['target'],device=device)
    ensembles={j:[] for j in a['members']};meta={name:value for name,value in a.items() if name not in ('fp','desc')}
    import joblib
    joblib.dump(meta,folder/'design.joblib',compress=3)
    for seed in SEEDS:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
        net=Net(graphs[0]['x'].shape[1],graphs[0]['e'].shape[1],desc.shape[1],len(d['tasks']),c).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.025);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,c['steps'],eta_min=.000025)
        order=np.arange(len(d['tasks']))
        for step in range(c['steps']):
            net.train()
            if step%len(order)==0:rng.shuffle(order)
            j=int(order[step%len(order)]);rows=a['members'][j];ix=rng.choice(rows,min(96,len(rows)),replace=False);ids=d['mi'][ix]
            p=net(*graph.batch(graphs,ids,device),fp[ids],desc[ids],j);z=loss(d['metrics'][j],p,target[ix]);assert torch.isfinite(z)
            opt.zero_grad(set_to_none=True);z.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();schedule.step()
            if step%1000==0:print(json.dumps(dict(stage='updates',part=part,seed=seed,step=step,loss=float(z.detach()))),flush=True)
        mp=folder/f'model_{seed}.pt';torch.save(dict(signature=sig(),state=net.state_dict(),config=c,ad=graphs[0]['x'].shape[1],ed=graphs[0]['e'].shape[1],dd=desc.shape[1],tasks=len(d['tasks'])),mp)
        replay=Net(graphs[0]['x'].shape[1],graphs[0]['e'].shape[1],desc.shape[1],len(d['tasks']),c).to(device);replay.load_state_dict(torch.load(mp,weights_only=True)['state'])
        for j in a['members']:
            raw=infer(net,d,a,graphs,fp,desc,j,device);np.testing.assert_array_equal(raw,infer(replay,d,a,graphs,fp,desc,j,device));ensembles[j].append(raw)
        del net,replay,opt
    saved=dict(signature=sig(),source=a['source'])
    for j,raw in ensembles.items():
        state=a['states'][j];p=np.mean(raw,axis=0);p=expit(p) if d['metrics'][j] in ('AUROC','AUPRC') else p*state['scale']+state['center']
        if d['tasks'][j]=='ppbr_az':p=p.clip(0,100)
        saved[f'p{j}']=p;saved[f'query{j}']=a['queries'][j]
    np.savez_compressed(path,**saved);del fp,desc,target;torch.cuda.empty_cache()
    print(json.dumps(dict(stage='train_shared_complete',part=part,seconds=round(time.monotonic()-start),source_rows=len(a['source']))),flush=True)
    return saved[f'p{k}']

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    runner.run(sys.modules[__name__],v.load())
if __name__=='__main__':main()
