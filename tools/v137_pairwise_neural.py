"""Endpoint-isolated neural rank/property experts with masked measured context."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v131_current_controls as v
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v137';SEEDS=(1374026,1374027);STEPS=1100;WEIGHTS=(.35,.75)
def sig():return hashlib.sha256((b.sha(__file__)+v.sig()).encode()).hexdigest()
def design(d,k,part):
    fit,q,x,qx,source=v.features(d,k,part,'augmented');n=len(fit)
    med=np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0. for z in x.T],np.float32)
    a=np.where(np.isfinite(x),x,med);scale=a.std(0).clip(.05);center=a.mean(0)
    desc=np.clip((np.r_[a,np.where(np.isfinite(qx),qx,med)]-center)/scale,-5,5).astype(np.float32)
    fp=np.sqrt(d['views']['multiview'][d['mi'][np.r_[fit,q]],:12288]).astype(np.float32)
    y=d['y'][fit];m=d['metrics'][k]
    yc=float(np.median(y)) if m=='MAE' else 0.;ys=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if m=='MAE' else 1.
    target=(y-yc)/ys if m=='MAE' else r83.ranknorm(y) if m=='Spearman' else y
    return dict(fit=fit,query=q,desc=desc,fp=fp,target=target.astype(np.float32),center=yc,scale=ys,source=source,median=med,feature_center=center,feature_scale=scale)
class Net(nn.Module):
    def __init__(self,dim):
        super().__init__();self.views=nn.ModuleList(nn.Sequential(nn.Linear(2048,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.3)) for _ in range(6))
        self.desc=nn.Sequential(nn.Linear(dim,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.3))
        self.head=nn.Sequential(nn.Linear(416,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.4),nn.Linear(128,1))
    def forward(self,fp,desc):return self.head(torch.cat([layer(fp[:,i*2048:(i+1)*2048]) for i,layer in enumerate(self.views)]+[self.desc(desc)],1)).squeeze(1)
def infer(net,a,fp,desc):
    net.eval();out=[]
    with torch.inference_mode():
        for lo in range(len(a['fit']),len(a['desc']),256):out.append(net(fp[lo:lo+256],desc[lo:lo+256]).cpu().numpy())
    return np.concatenate(out).astype(float)
def predict(d,k,part,c=None):
    folder=OUT/'models'/d['tasks'][k]/part;file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);return z['p'].copy()
    a=design(d,k,part);folder.mkdir(parents=True,exist_ok=True);start=time.monotonic();fp=torch.as_tensor(a['fp'],device='cuda');desc=torch.as_tensor(a['desc'],device='cuda');target=torch.as_tensor(a['target'],device='cuda');members=[];m=d['metrics'][k]
    for seed in SEEDS:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);net=Net(desc.shape[1]).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.03);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00003)
        for step in range(STEPS):
            net.train();ix=rng.choice(len(a['fit']),128,replace=len(a['fit'])<128);p=net(fp[ix],desc[ix]);y=target[ix]
            if m in ('AUROC','AUPRC'):
                loss=nn.functional.binary_cross_entropy_with_logits(p,y);positive=p[y>.5];negative=p[y<.5]
                if len(positive) and len(negative):loss=loss+.25*nn.functional.softplus(negative[:,None]-positive[None]).mean()
            elif m=='MAE':loss=(p-y).abs().mean()
            else:
                delta=y[:,None]-y[None];mask=delta.abs()>.1
                pair=nn.functional.softplus(-delta.sign()*(p[:,None]-p[None]))
                loss=.5*(p-y).square().mean()+.5*pair[mask].mean()
            loss=loss+.001*p.square().mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();schedule.step()
        raw=infer(net,a,fp,desc);mp=folder/f'model_{seed}.pt';torch.save(dict(signature=sig(),weights={n:z.detach().cpu() for n,z in net.state_dict().items()},dim=desc.shape[1]),mp)
        replay=Net(desc.shape[1]).cuda();replay.load_state_dict(torch.load(mp,weights_only=True)['weights']);np.testing.assert_array_equal(raw,infer(replay,a,fp,desc));members.append(raw);del net,replay,opt
    raw=np.mean(members,axis=0);p=expit(raw) if m in ('AUROC','AUPRC') else raw*a['scale']+a['center']
    if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
    np.savez_compressed(file,signature=sig(),fit=a['fit'],query=a['query'],p=p,center=a['center'],scale=a['scale'])
    del fp,desc,target;torch.cuda.empty_cache();print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,seconds=round(time.monotonic()-start))),flush=True);return p
def record(d,k,part,w):
    q=v.split(d,k,part)[1];m=d['metrics'][k];a=v.baseline(d,k,part);z=predict(d,k,part);p=v.mix(m,a,z,w)
    return dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],a),expert=r83.score(m,d['y'][q],z),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],a,p))
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    d=v.load();OUT.mkdir(exist_ok=True);doc=dict(signature=sig(),tasks=v.tasks(d),seeds=SEEDS,steps=STEPS,weights=WEIGHTS,
        model='Endpoint-isolated six-view fingerprint neural network + augmented chemical/shape/pharmacophore and target-excluded measured context. Pairwise discrimination auxiliary; MAE for MAE.',
        selection='Two blend weights only on fold0/1, freeze all tasks before fold2/valid; all4 positive and mean above threshold; no fallback.',test_labels_used=False,external_data=False,baseline='v130 with historical OOF proxy caveats; labels reused.')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(doc))
    else:b.save(path,doc)
    development=[];selected=[]
    for task in v.tasks(d):
        k=d['tasks'].index(task);choices=[]
        for w in WEIGHTS:
            parts=[record(d,k,p,w) for p in ('fold0','fold1')];choices.append(dict(task=task,metric=d['metrics'][k],config={},weight=w,parts=parts,objective=min(r['gain'] for r in parts)))
        best=max(choices,key=lambda r:r['objective']);development.extend(choices)
        if best['objective']>v.threshold(d['metrics'][k]):selected.append(best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=development));print(json.dumps(dict(stage='development',best=best)),flush=True)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=development));b.save(OUT/'frozen_selection.json',dict(signature=sig(),selected=selected));records=[]
    for chosen in selected:
        k=d['tasks'].index(chosen['task']);parts=[record(d,k,p,chosen['weight']) for p in ('fold0','fold1','fold2','valid')]
        accepted=bool(min(r['gain'] for r in parts)>0 and np.mean([r['gain'] for r in parts])>v.threshold(d['metrics'][k]))
        records.append(dict(**chosen,confirmation=parts,accepted=accepted));print(json.dumps(dict(stage='confirmation',**records[-1])),flush=True)
        if accepted:predict(d,k,'final')
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']]))
if __name__=='__main__':main()
