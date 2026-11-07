"""Report-only fixed-config removal of 3D; never used to retune selection."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v100_shape_kernel as v

def no3d(d,k,part,cfg):
    a=v.state(d,k,part);fit,q=a['fit'],a['query'];fi,qi=d['mi'][fit],d['mi'][q]
    gr=torch.zeros((len(fi),len(fi)),device='cuda');gq=torch.zeros((len(qi),len(fi)),device='cuda')
    raw=d['views']['multiview'][:,:12288]
    for start in range(0,12288,2048):
        tx=torch.as_tensor((raw[fi,start:start+2048]>0).astype(np.float32),device='cuda');tn=tx.sum(1)
        for rows,dest in ((fi,gr),(qi,gq)):
            for lo in range(0,len(rows),512):
                x=torch.as_tensor((raw[rows[lo:lo+512],start:start+2048]>0).astype(np.float32),device='cuda');dot=x@tx.T
                tan=(dot/(x.sum(1)[:,None]+tn[None]-dot).clamp_min(1e-8)).clamp(0,1)
                dest[lo:lo+len(x)]+=(.5*tan+.5*tan.square())/6
    tr,qe,_,_,active=v.normalized(d['views']['physchem'],fi,qi);sr=torch.zeros_like(gr);sq=torch.zeros_like(gq)
    tx=torch.as_tensor(tr,device='cuda');tn=tx.square().sum(1)
    for src,dest in ((tr,sr),(qe,sq)):
        for lo in range(0,len(src),512):
            x=torch.as_tensor(src[lo:lo+512],device='cuda');dist=(x.square().sum(1)[:,None]+tn[None]-2*x@tx.T).clamp_min(0)/len(active)
            dest[lo:lo+len(x)]=.5*torch.exp(-.5*dist)+.5*torch.exp(-2*dist)
    sr.diagonal().fill_(1.)
    mat=.5*gr+.5*sr if cfg['view']=='sum' else gr*sr
    query=.5*gq+.5*sq if cfg['view']=='sum' else gq*sq
    mat=(mat+mat.T)*.5;mat.diagonal().add_(cfg['alpha']);coef=torch.cholesky_solve(torch.as_tensor(a['target'],dtype=torch.float32,device='cuda'),torch.linalg.cholesky(mat))
    value=(query@coef).cpu().numpy()[:,('direct','residual').index(cfg['mode'])];m=d['metrics'][k]
    if cfg['mode']=='direct':p=value*float(a['scale'])+float(a['center'])
    elif m in ('AUROC','AUPRC'):p=expit(a['qoff'])+value
    else:p=a['qoff']+value*float(a['scale'])
    return expit(4*(p-.5)) if m in ('AUROC','AUPRC') else p

def main():
    d=v.load();conf=json.loads((v.OUT/'confirmation.json').read_text());assert conf['status']=='COMPLETE';records=[]
    for r in conf['records']:
        if not r['accepted']:continue
        k=d['tasks'].index(r['task']);m=r['metric']
        for part in ('fold0','fold1','fold2','valid'):
            q=v.r83.split(d,k,part)[1];base=v.base(d,k,part)
            full=v.mix(m,base,v.expert(d,k,part,r['config']),r['weight'])
            ablated=v.mix(m,base,no3d(d,k,part,r['config']),r['weight'])
            rec=dict(task=r['task'],part=part,config=r['config'],weight=r['weight'],baseline=v.r83.score(m,d['y'][q],base),
                with3d=v.r83.score(m,d['y'][q],full),without3d=v.r83.score(m,d['y'][q],ablated),
                gain_of_3d=v.r83.gain(m,d['y'][q],ablated,full))
            records.append(rec);v.emit(stage='ablation_report_only',**rec)
    v.b.save(v.OUT/'ablation.json',dict(status='COMPLETE',selection_effect='NONE_REPORT_ONLY',records=records,
        limitations='Fixed selected hyperparameters favor full model. Removing 3D is not separately tuned. Historical labels/proxy are reused.'))

if __name__=='__main__':
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');main()
