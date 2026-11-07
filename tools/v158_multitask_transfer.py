"""Endpoint-focused pure-structure multitask encoders with no assay covariates."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v158';SEEDS=(1584026,1584027);STEPS=1800
CONFIGS=(dict(auxiliary=.3),)
METHOD='Target-focused structure-only shared multitask encoder, auxiliary task-specific heads, balanced sampling and gradient projection to reduce negative transfer. No assay values or historical predictions as inputs. Query scaffolds excluded from all auxiliary task fitting labels. Rank+pairwise Spearman, L1 MAE, BCE+ranking classification. Two seeds, fixed duration.'

def sig():return hashlib.sha256((b.sha(__file__)+v.sig()+b.sha(runner.__file__)).encode()).hexdigest()
def key(c):return 'aux'+str(c['auxiliary'])

def design(d,k,part):
    fit,q=v.split(d,k,part);limit=2 if part=='final' else 1
    other=np.flatnonzero((d['tid']!=k)&(d['split']<limit))
    if part!='final':other=other[~np.isin(d['groups'][other],d['groups'][q])]
    source=np.r_[fit,other];assert not set(q)&set(source) and not np.any(d['tid'][other]==k)
    if part!='final':assert not set(d['groups'][source])&set(d['groups'][q])
    unique=np.unique(d['mi'][source]);raw=np.c_[d['views']['physchem'],d['conformer_features'],d['pharmacophore']]
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in raw[unique].T],np.float32);x=np.where(np.isfinite(raw),raw,med);center=x[unique].mean(0);scale=x[unique].std(0).clip(.05)
    desc=np.clip((x-center)/scale,-5,5).astype(np.float32);fp=np.sqrt(d['views']['multiview'][:,:12288]).astype(np.float32)
    target=np.zeros(len(d['y']),np.float32);states={};members={}
    for j,m in enumerate(d['metrics']):
        ix=source[d['tid'][source]==j]
        if not len(ix):continue
        y=d['y'][ix];yc=float(np.median(y)) if m=='MAE' else 0.;ys=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if m=='MAE' else 1.
        target[ix]=(y-yc)/ys if m=='MAE' else r83.ranknorm(y) if m=='Spearman' else y;states[j]=dict(center=yc,scale=ys);members[j]=ix
    return dict(fit=fit,query=q,source=source,other=other,desc=desc,fp=fp,target=target,states=states,members=members,median=med,feature_center=center,feature_scale=scale)

class Net(nn.Module):
    def __init__(self,dim,tasks):
        super().__init__();self.views=nn.ModuleList(nn.Sequential(nn.Linear(2048,32),nn.LayerNorm(32),nn.SiLU(),nn.Dropout(.25)) for _ in range(6));self.desc=nn.Sequential(nn.Linear(dim,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.2));self.shared=nn.Sequential(nn.Linear(256,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.25));self.heads=nn.ModuleList(nn.Sequential(nn.Linear(128,48),nn.SiLU(),nn.Dropout(.2),nn.Linear(48,1)) for _ in range(tasks))
    def forward(self,fp,desc,k):
        h=self.shared(torch.cat([layer(fp[:,j*2048:(j+1)*2048]) for j,layer in enumerate(self.views)]+[self.desc(desc)],1));return self.heads[k](h).squeeze(1)

def lossfn(m,p,y):
    if m in ('AUROC','AUPRC'):
        loss=nn.functional.binary_cross_entropy_with_logits(p,y);a,z=p[y>.5],p[y<.5]
        if len(a) and len(z):loss=loss+.2*nn.functional.softplus(z[:,None]-a[None]).mean()
    elif m=='MAE':loss=(p-y).abs().mean()
    else:
        delta=y[:,None]-y[None];mask=delta.abs()>.1;loss=.5*(p-y).square().mean()
        if mask.any():loss=loss+.5*nn.functional.softplus(-delta.sign()*(p[:,None]-p[None]))[mask].mean()
    return loss+.001*p.square().mean()

def infer(net,d,a,fp,desc,k):
    net.eval();out=[]
    with torch.inference_mode():
        for lo in range(0,len(a['query']),256):
            ix=d['mi'][a['query'][lo:lo+256]];out.append(net(fp[ix],desc[ix],k).cpu().numpy())
    return np.concatenate(out).astype(float)

def predict(d,k,part,c):
    folder=OUT/'models'/d['tasks'][k]/part/key(c);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);return z['p'].copy()
    start=time.monotonic();a=design(d,k,part);folder.mkdir(parents=True,exist_ok=True);fp=torch.as_tensor(a['fp'],device='cuda');desc=torch.as_tensor(a['desc'],device='cuda');target=torch.as_tensor(a['target'],device='cuda');members=[];m=d['metrics'][k]
    for seed in SEEDS:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);net=Net(desc.shape[1],len(d['tasks'])).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.03);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00003)
        parameters=list(net.parameters());shared=[p for name,p in net.named_parameters() if not name.startswith('heads.')];other_tasks=[j for j in a['members'] if j!=k]
        for step in range(STEPS):
            net.train();ix=rng.choice(a['fit'],128,replace=len(a['fit'])<128);pred=net(fp[d['mi'][ix]],desc[d['mi'][ix]],k);mainloss=lossfn(m,pred,target[ix]);opt.zero_grad(set_to_none=True);mainloss.backward()
            # Project conflicting auxiliary shared gradients off the main direction.
            main_grad={id(p):p.grad.clone() for p in shared if p.grad is not None};saved={id(p):None if p.grad is None else p.grad.clone() for p in parameters};opt.zero_grad(set_to_none=True)
            j=other_tasks[step%len(other_tasks)];rows=a['members'][j];ix=rng.choice(rows,128,replace=len(rows)<128);auxloss=lossfn(d['metrics'][j],net(fp[d['mi'][ix]],desc[d['mi'][ix]],j),target[ix]);auxloss.backward()
            dot=sum((main_grad[id(p)]*p.grad).sum() for p in shared if id(p) in main_grad and p.grad is not None);norm=sum(z.square().sum() for z in main_grad.values()).clamp_min(1e-12);ratio=torch.minimum(dot/norm,torch.zeros_like(dot))
            for p in parameters:
                ag=p.grad
                if ag is not None:
                    if id(p) in main_grad:ag=ag-ratio*main_grad[id(p)]
                    ag=ag*c['auxiliary']
                mg=saved[id(p)];p.grad=ag if mg is None else mg if ag is None else mg+ag
            nn.utils.clip_grad_norm_(parameters,2.);opt.step();schedule.step()
        raw=infer(net,d,a,fp,desc,k);mp=folder/f'model_{seed}.pt';torch.save(dict(signature=sig(),weights={n:p.detach().cpu() for n,p in net.state_dict().items()},dim=desc.shape[1],tasks=len(d['tasks'])),mp)
        replay=Net(desc.shape[1],len(d['tasks'])).cuda();replay.load_state_dict(torch.load(mp,weights_only=True)['weights']);np.testing.assert_array_equal(raw,infer(replay,d,a,fp,desc,k));members.append(raw);del net,replay,opt
    state=a['states'][k];p=np.mean(members,axis=0);p=expit(p) if m in ('AUROC','AUPRC') else p*state['scale']+state['center']
    if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
    np.savez_compressed(file,signature=sig(),fit=a['fit'],query=a['query'],source=a['source'],p=p,center=state['center'],scale=state['scale']);del fp,desc,target;torch.cuda.empty_cache();print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,seconds=round(time.monotonic()-start))),flush=True);return p

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');runner.run(sys.modules[__name__],v.load())
if __name__=='__main__':main()
