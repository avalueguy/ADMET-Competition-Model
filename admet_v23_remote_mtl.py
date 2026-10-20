"""Remote v23 multi-task feature model.

This script is intentionally self-contained: it uses RDKit Morgan fingerprints,
robust scalar descriptors and a shared PyTorch trunk with task-specific heads.
It never touches v20/v21/v22 outputs and writes only admet_model_outputs_v23 and
admet_submissions/v23 on the authorized remote host.
"""
from __future__ import annotations
import csv, hashlib, io, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from scipy.stats import spearmanr

ROOT=Path('/home/czp123/admet_v23_20260928')
DATA=ROOT/'admet_competition_data/data_benchmark'
OUT=ROOT/'admet_model_outputs_v23'
SUB=ROOT/'admet_submissions/v23'
RDLogger.DisableLog('rdApp.*')
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
SEED=230927
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
TASKS=['ames','bbb_martins','bioavailability_ma','caco2_wang','clearance_hepatocyte_az','clearance_microsome_az','cyp2c9_substrate_carbonmangels','cyp2c9_veith','cyp2d6_substrate_carbonmangels','cyp2d6_veith','cyp3a4_substrate_carbonmangels','cyp3a4_veith','dili','half_life_obach','herg','hia_hou','ld50_zhu','lipophilicity_astrazeneca','pgp_broccatelli','ppbr_az','solubility_aqsoldb','vdss_lombardo']
METRICS={t:m for t,m in zip(pd.read_csv(ROOT/'admet_official/submission_template.csv',dtype=str).drop_duplicates('task').task, pd.read_csv(ROOT/'admet_official/submission_template.csv',dtype=str).drop_duplicates('task').metric)}
CLS={t: METRICS[t] in ('AUROC','AUPRC') for t in TASKS}
GEN=GetMorganGenerator(radius=2,fpSize=1024,includeChirality=True)
DESC=[Descriptors.MolWt,Descriptors.MolLogP,Descriptors.TPSA,Descriptors.NumHDonors,Descriptors.NumHAcceptors,Descriptors.NumRotatableBonds,Descriptors.RingCount,Descriptors.FractionCSP3,Descriptors.HeavyAtomCount,Descriptors.NHOHCount,Descriptors.NOCount,Descriptors.NumAromaticRings,Descriptors.NumAliphaticRings,Descriptors.NumSaturatedRings,Descriptors.NumHeteroatoms,Descriptors.LabuteASA]

def feat(task, split):
    p=OUT/'features'/f'{task}_{split}.npz'; p.parent.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA/task/f'{split}.csv')
    key=hashlib.sha256((task+'|'+split+'|'+ '|'.join(df.Drug.astype(str))).encode()).hexdigest()
    if p.exists():
        z=np.load(p); assert str(z['key'])==key
        return z['x'], df.Y.to_numpy(float) if 'Y' in df else None, df
    x=np.zeros((len(df),1040),np.float32)
    for i,s in enumerate(df.Drug.astype(str)):
        mol=Chem.MolFromSmiles(s)
        if mol is None: continue
        try: x[i,:1024]=GEN.GetFingerprintAsNumPy(mol)
        except Exception: pass
        vals=[]
        for fn in DESC:
            try: v=float(fn(mol))
            except Exception: v=np.nan
            vals.append(v if np.isfinite(v) and abs(v)<1e8 else np.nan)
        x[i,1024:]=np.nan_to_num(vals,nan=0.0,posinf=0.0,neginf=0.0)
    np.savez_compressed(p,x=x,key=key)
    return x, df.Y.to_numpy(float) if 'Y' in df else None, df

class SharedMTL(nn.Module):
    def __init__(self,n_tasks):
        super().__init__(); self.back=nn.Sequential(nn.Linear(1040,384),nn.LayerNorm(384),nn.SiLU(),nn.Dropout(.12),nn.Linear(384,192),nn.LayerNorm(192),nn.SiLU(),nn.Dropout(.10)); self.heads=nn.ModuleList([nn.Sequential(nn.Linear(192,64),nn.SiLU(),nn.Dropout(.10),nn.Linear(64,1)) for _ in range(n_tasks)])
    def forward(self,x,k): return self.heads[k](self.back(x)).squeeze(-1)

def prep(x):
    # fit-independent transforms; descriptors occupy the final 16 dimensions
    x=x.copy(); d=x[:,1024:]; med=np.nanmedian(d,axis=0); d=np.nan_to_num(d,nan=med); mu=d.mean(0); sd=d.std(0); sd[sd<1e-6]=1; x[:,1024:]=np.clip((d-mu)/sd,-6,6); x[:,:1024]=np.log1p(x[:,:1024]); return x.astype('float32')

def loss_fn(pred,y,metric):
    if metric in ('AUROC','AUPRC'):
        pos=max(float(y.mean()),.01); pw=min(4.,max(1.,((1-pos)/pos)**.5)); return nn.functional.binary_cross_entropy_with_logits(pred,y,pos_weight=torch.tensor(pw,device=pred.device))
    if metric=='MAE': return nn.functional.smooth_l1_loss(pred,y,beta=.25)
    return nn.functional.smooth_l1_loss(pred,y,beta=.35)

def train():
    OUT.mkdir(exist_ok=True); (OUT/'features').mkdir(exist_ok=True)
    data=[]
    for t in TASKS:
        x,y,df=feat(t,'train'); xv,yv,dv=feat(t,'valid'); xt,_,dt=feat(t,'test');
        # task-local robust scaling for continuous labels
        if not CLS[t]:
            med=float(np.median(y)); sc=float(np.quantile(y,.75)-np.quantile(y,.25)); sc=max(sc,1e-3); yy=(y-med)/sc
        else: med,sc=0.,1.; yy=y
        data.append(dict(task=t,x=torch.from_numpy(prep(x)),y=torch.from_numpy(yy.astype('float32')),valid=torch.from_numpy(prep(xv)),yv=yv,test=torch.from_numpy(prep(xt)),med=med,sc=sc,df=df,dv=dv,dt=dt))
        print(json.dumps({'stage':'features','task':t,'train':len(x),'valid':len(xv),'test':len(xt)}),flush=True)
    model=SharedMTL(len(TASKS)).to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=.012); rng=np.random.default_rng(SEED)
    # Balanced task sampling prevents large Veith/AqSolDB datasets from dominating.
    steps=1600; batch=512
    model.train()
    for step in range(steps):
        k=int(rng.integers(len(data))); z=data[k]; n=len(z['x']); ii=rng.integers(n,size=min(batch,n)); xb=z['x'][ii].to(DEVICE); yb=z['y'][ii].to(DEVICE); p=model(xb,k); l=loss_fn(p,yb,METRICS[z['task']]); opt.zero_grad(set_to_none=True); l.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step()
        if step%100==0: print(json.dumps({'stage':'train','step':step,'task':z['task'],'loss':float(l.detach())}),flush=True)
    model.eval(); torch.save(model.state_dict(),OUT/'mtl.pt')
    preds={}
    with torch.inference_mode():
        for k,z in enumerate(data):
            (OUT/z['task']).mkdir(parents=True, exist_ok=True)
            def run(xx):
                q=[]
                for i in range(0,len(xx),2048): q.append(model(xx[i:i+2048].to(DEVICE),k).cpu().numpy())
                q=np.concatenate(q)
                if CLS[z['task']]: q=1/(1+np.exp(-q))
                else: q=q*z['sc']+z['med']
                return q
            pv=run(z['valid']); pt=run(z['test']);
            pd.DataFrame({'Y_true':z['yv'],'Y_pred':pv}).to_csv(OUT/z['task']/'valid.csv',index=False); pd.DataFrame({'Y_pred':pt}).to_csv(OUT/z['task']/'test.csv',index=False); preds[z['task']]=pt
            print(json.dumps({'stage':'predict','task':z['task']}),flush=True)
    release(preds)

def metric_value(metric,y,p):
    try:
        if metric=='MAE': return -mean_absolute_error(y,p)
        if metric=='Spearman': return float(spearmanr(y,p).statistic)
        if metric=='AUROC': return roc_auc_score(y,p)
        return average_precision_score(y,p)
    except Exception:return -1e9

def release(preds):
    base=next((ROOT/'admet_submissions/v20').glob('*.csv')); rows=list(csv.reader(io.StringIO(base.read_bytes().decode('utf-8-sig'),newline=''))); templ=list(csv.reader(io.StringIO((ROOT/'admet_official/submission_template.csv').read_bytes().decode('utf-8-sig'),newline='')))
    changed={}; report={}
    for k,t in enumerate(TASKS):
        # Select alpha against the fixed official valid baseline from v16; this is a gate, not a claim about platform score.
        vp=OUT/t/'valid.csv'; v16=ROOT/'admet_release_v16'/t/'validation_predictions.csv'
        alpha=0.0
        if vp.exists() and v16.exists():
            a=pd.read_csv(vp).Y_pred.to_numpy(float); b=pd.read_csv(v16).Y_pred.to_numpy(float); y=pd.read_csv(vp).Y_true.to_numpy(float)
            best=metric_value(METRICS[t],y,b)
            for aa in np.linspace(.05,.5,10):
                q=(1-aa)*b+aa*a; val=metric_value(METRICS[t],y,q)
                if val>best+1e-5: best=val; alpha=float(aa)
        if alpha>0:
            # v20 remains anchor; use rank blend for classification and raw blend for regression.
            q=preds[t]; old=np.array([float(r[4]) for r in rows[1:] if r[0]==t]);
            if CLS[t]:
                rq=(rankdata(q,method='average')-.5)/len(q); ro=(rankdata(old,method='average')-.5)/len(old); q=(1-alpha)*ro+alpha*rq
            else: q=(1-alpha)*old+alpha*q
            changed[t]=int(len(q)); report[t]={'alpha':alpha,'valid_metric_gate':True}
            j=0
            for r in rows[1:]:
                if r[0]==t: r[4]=repr(float(np.clip(q[j],0,1) if CLS[t] else q[j])); j+=1
    assert len(rows)==len(templ) and rows[0]==templ[0]
    for a,b in zip(rows[1:],templ[1:]): assert a[:4]+a[5:]==b[:4]+b[5:]
    SUB.mkdir(parents=True,exist_ok=True); out=SUB/'我从山北来带着兰花草-小分子药物预测技术创新赛-20260928-v23.csv'; buf=io.StringIO(newline=''); csv.writer(buf,lineterminator='\r\n').writerows(rows); out.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode())
    info={'status':'V23_REMOTE_MTL','rows':len(rows)-1,'columns':6,'tasks':len(TASKS),'changed_tasks':sorted(changed),'changed_rows':sum(changed.values()),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'baseline':'v20','baseline_score':50.019727,'gate':report,'platform_score':None}
    (SUB/'verification.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(info,ensure_ascii=False),flush=True)

if __name__=='__main__': train()
