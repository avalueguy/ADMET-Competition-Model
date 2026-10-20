"""Remote supervised MoLFormer tail fine-tuning with cached label-free prefixes."""
import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit
from rdkit import Chem, RDLogger
from transformers import AutoModel, AutoTokenizer

from admet_v16_models import load_task
from admet_v6_models import Preprocess
from admet_v8_models import RobustTarget
from admet_v12_models import inner_split
from admet_v6_runner import remote_only, partition
from admet_v20_residual import read_json, save_json, sha, metric_for, gain, utility
from admet_v21_anchor import anchor_oof, anchor_valid, anchor_test
from admet_v16_ensemble import combine

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'admet_model_outputs_v21'
PROTOCOL = ROOT/'admet_v21_protocol.json'
MODEL = ROOT/'admet_pretrained/molformer'
SOURCE = Path(__file__).resolve()
BASE = None
TOKENIZER = None


def emit(stage, **kw):
    print(json.dumps(dict(stage=stage, **kw)), flush=True)


def pretrained():
    global BASE, TOKENIZER
    if BASE is None:
        torch.manual_seed(21200)
        provenance = read_json(MODEL/'provenance.json')
        assert provenance['revision']==read_json(PROTOCOL)['pretrained_revision']
        for record in provenance['files']:
            assert sha(MODEL/record['name'])==record['sha256']
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
        BASE = AutoModel.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True,
                                         deterministic_eval=True).eval()
        BASE.requires_grad_(False)
    return BASE, TOKENIZER


def text_views(smiles):
    canonical, random = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError('Invalid molecule encountered; no silent replacement')
        cs = Chem.MolToSmiles(mol, isomericSmiles=False)
        clean = Chem.MolFromSmiles(cs)
        seed = int(hashlib.sha256(cs.encode()).hexdigest()[:7],16)+1
        canonical.append(cs)
        random.append(Chem.MolToRandomSmilesVect(clean,1,randomSeed=seed,isomericSmiles=False)[0])
    return [canonical, random]


def cache_prefix(task,d):
    p = read_json(PROTOCOL)
    path = OUT/'prefix'/f'{task}.pt'
    key = hashlib.sha256(json.dumps([d['smiles'],p['pretrained_revision'],p['max_tokens'],
                                    p['prefix_layers_frozen'],p['views']]).encode()).hexdigest()
    if path.exists():
        cached = torch.load(path,map_location='cpu',weights_only=False)
        assert cached['key']==key
        return cached['views']
    base,tok = pretrained()
    base.cuda().eval()
    views = []
    truncated = []
    started = time.time()
    for text in text_views(d['smiles']):
        count=sum(len(ids)>p['max_tokens'] for ids in tok(text,truncation=False)['input_ids'])
        truncated.append(count)
        tokens=tok(text,padding=False,truncation=True,max_length=p['max_tokens'])
        order=sorted(range(len(text)),key=lambda i:len(tokens['input_ids'][i]))
        values=[None]*len(text)
        with torch.inference_mode():
            for start in range(0,len(order),32):
                ix=order[start:start+32]
                raw=[{k:tokens[k][int(i)] for k in tokens} for i in ix]
                inp={k:v.cuda() for k,v in tok.pad(raw,padding=True,return_tensors='pt').items()}
                mask=base.get_extended_attention_mask(inp['attention_mask'],inp['input_ids'].shape)
                pos=torch.arange(inp['input_ids'].shape[1],device='cuda').unsqueeze(0)
                # Float32 frozen prefix ensures the cache is portable between fits.
                h=base.embeddings(input_ids=inp['input_ids'])
                for layer in base.encoder.layer[:p['prefix_layers_frozen']]:
                    h=layer(h,attention_mask=mask,position_ids=pos)[0]
                for j,i in enumerate(ix):
                    length=int(inp['attention_mask'][j].sum())
                    values[i]=h[j,:length].half().cpu().clone()
        views.append(values)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(key=key,views=views),path)
    base.cpu()
    torch.cuda.empty_cache()
    save_json(path.with_suffix('.json'),dict(task=task,rows=len(d['smiles']),variants=2,
              labels_used=False,prefix_layers=p['prefix_layers_frozen'],max_tokens=p['max_tokens'],
              truncated_rows_per_view=truncated,seconds=time.time()-started,
              cache_sha256=sha(path),pretrained_revision=p['pretrained_revision']))
    emit('prefix_cached',task=task,seconds=round(time.time()-started),bytes=path.stat().st_size)
    return views


class TailModel(nn.Module):
    def __init__(self,extra_dim):
        super().__init__()
        base,_=pretrained()
        p=read_json(PROTOCOL)
        self.layers=copy.deepcopy(base.encoder.layer[p['prefix_layers_frozen']:])
        self.norm=copy.deepcopy(base.LayerNorm)
        self.layers.requires_grad_(True)
        self.norm.requires_grad_(True)
        # Rotary buffers are non-persistent. Allocate the full token range in
        # float32 before autocast so replay does not depend on earlier batches.
        for layer in self.layers:
            layer.attention.self.rotary_embeddings._set_cos_sin_cache(
                p['max_tokens'], device=torch.device('cpu'), dtype=torch.float32)
        self.extra=nn.Sequential(nn.Linear(extra_dim,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.2))
        self.head=nn.Sequential(nn.Linear(896,192),nn.LayerNorm(192),nn.GELU(),nn.Dropout(p['head_dropout']),nn.Linear(192,1))

    def train(self,mode=True):
        super().train(mode)
        # Fix random feature projections; only supervised weights and dropout vary.
        for layer in self.layers:
            layer.attention.self.feature_map.eval()
        return self

    def forward(self,h,mask,extra):
        ext=(1-mask[:,None,None,:].to(h.dtype))*torch.finfo(h.dtype).min
        pos=torch.arange(h.shape[1],device=h.device).unsqueeze(0)
        for layer in self.layers:
            h=layer(h,attention_mask=ext,position_ids=pos)[0]
        h=self.norm(h)
        pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1).unsqueeze(-1).clamp_min(1)
        return self.head(torch.cat([pooled,self.extra(extra)],dim=1)).squeeze(-1)


def batch(prefix,ix,variant):
    parts=[prefix[variant][int(i)] for i in ix]
    lengths=torch.tensor([len(a) for a in parts],device='cuda')
    h=nn.utils.rnn.pad_sequence(parts,batch_first=True).cuda().float()
    mask=torch.arange(h.shape[1],device='cuda')[None,:]<lengths[:,None]
    return h,mask


def predict(net,prefix,rows,extra,tf,variants=(0,1)):
    net.eval()
    values=[]
    with torch.inference_mode():
        for variant in variants:
            parts=[]
            for start in range(0,len(rows),64):
                h,mask=batch(prefix,rows[start:start+64],variant)
                q=net(h,mask,torch.tensor(extra[start:start+64],device='cuda')).float()
                parts.append(q.cpu().numpy())
            p=np.concatenate(parts)
            values.append(expit(p) if tf is None else tf.decode(p))
    q=np.mean(values,axis=0)
    assert np.isfinite(q).all()
    return q


def loss_fn(pred,y,metric):
    if metric in ['AUPRC','AUROC']:
        loss=nn.functional.binary_cross_entropy_with_logits(pred,y)
        pos,neg=pred[y>.5],pred[y<.5]
        if len(pos) and len(neg):
            loss=loss+.15*nn.functional.softplus(neg[None,:]-pos[:,None]).mean()
        return loss
    if metric=='MAE':
        return nn.functional.l1_loss(pred,y)
    loss=nn.functional.smooth_l1_loss(pred,y,beta=.3)
    delta=y[:,None]-y[None,:]
    mask=delta.abs()>.15
    if mask.any():
        diff=pred[:,None]-pred[None,:]
        loss=loss+.15*nn.functional.softplus(-torch.sign(delta[mask])*diff[mask]).mean()
    return loss


def train_one(prefix,extra,fit,y,metric,seed,epochs,hold=None,hold_y=None):
    p=read_json(PROTOCOL)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    pre=Preprocess().fit(extra[fit])
    x=pre.transform(extra[fit])
    tf=None if metric in ['AUPRC','AUROC'] else RobustTarget(y,metric)
    target=y if tf is None else tf.encode(y)
    labels=torch.tensor(target,dtype=torch.float32,device='cuda')
    xx=torch.tensor(x,device='cuda')
    net=TailModel(x.shape[1]).cuda()
    opt=torch.optim.AdamW([{'params':list(net.layers.parameters())+list(net.norm.parameters()),'lr':p['encoder_lr']},
                           {'params':list(net.extra.parameters())+list(net.head.parameters()),'lr':p['head_lr']}],
                          weight_decay=p['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,p['epochs'],eta_min=1e-6)
    best=-np.inf
    chosen=2
    stale=0
    curves=[]
    for epoch in range(1,epochs+1):
        net.train()
        order=rng.permutation(len(fit))
        losses=[]
        for start in range(0,len(order),p['batch_size']):
            j=order[start:start+p['batch_size']]
            h,mask=batch(prefix,fit[j],(epoch-1)%2)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=net(h,mask,xx[j]).float()
                loss=loss_fn(prediction,labels[j],metric)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if hold is not None and epoch%2==0:
            q=predict(net,prefix,hold,pre.transform(extra[hold]),tf)
            score=utility(metric,hold_y,q)
            curves.append(dict(epoch=epoch,utility=float(score),loss=float(np.mean(losses))))
            if score>best+1e-6:
                best,chosen,stale=score,epoch,0
            else:
                stale+=1
            if stale>=4:
                break
    del labels,xx,opt,scheduler
    return net,pre,tf,chosen,curves


def materialize(task,d,prefix,extra,fit,query,stage):
    dest=OUT/task/stage
    dest.mkdir(parents=True,exist_ok=True)
    cache,meta=dest/'predictions.npz',dest/'metadata.json'
    if cache.exists() and meta.exists():
        record=read_json(meta)
        assert record['protocol_sha256']==sha(PROTOCOL)
        assert record['source_sha256']==sha(SOURCE)
        with np.load(cache) as z:
            np.testing.assert_array_equal(z['fit'],fit)
            np.testing.assert_array_equal(z['query'],query)
            return z['prediction'].copy()
    p=read_json(PROTOCOL)
    metric=metric_for(task)
    labels=pd.concat([d['frames']['train'],d['frames']['valid']],ignore_index=True).Y.to_numpy(float)
    assert np.max(fit)<len(labels) and not set(fit)&set(query)
    y=labels[fit]
    a,b,split_seed=inner_split(d,fit,y,metric)
    assert not set(d['groups'][fit[a]])&set(d['groups'][fit[b]])
    outputs=[]
    epochs=[]
    allcurves=[]
    started=time.time()
    for i,seed in enumerate(p['seeds']):
        net,_,_,niter,curve=train_one(prefix,extra,fit[a],y[a],metric,seed,p['epochs'],fit[b],y[b])
        del net
        torch.cuda.empty_cache()
        emit('inner_selected',task=task,partition=stage,seed=seed,epochs=niter)
        net,pre,tf,_,_=train_one(prefix,extra,fit,y,metric,seed,niter)
        q=predict(net,prefix,query,pre.transform(extra[query]),tf)
        checkpoint=dest/f'model_{i}.pt'
        torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},checkpoint)
        joblib.dump(dict(pre=pre,target=tf),dest/f'preprocess_{i}.joblib',compress=3)
        replay=TailModel(extra.shape[1]).cuda()
        replay.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        qr=predict(replay,prefix,query,pre.transform(extra[query]),tf)
        np.testing.assert_allclose(q,qr,rtol=1e-6,atol=1e-6)
        outputs.append(qr)
        epochs.append(niter)
        allcurves.append(curve)
        del net,replay
        torch.cuda.empty_cache()
    q=np.mean(outputs,axis=0)
    np.savez_compressed(cache,fit=fit,query=query,prediction=q)
    save_json(meta,dict(task=task,stage=stage,fit_rows=len(fit),query_rows=len(query),
              fit_target_sha256=hashlib.sha256(np.asarray(y,dtype='<f8').tobytes()).hexdigest(),
              protocol_sha256=sha(PROTOCOL),source_sha256=sha(SOURCE),inner_seed=split_seed,
              inner_scaffold_overlap=0,epochs=epochs,curves=allcurves,model_replay_passed=True,
              seconds=time.time()-started,files={p.name:sha(p) for p in dest.iterdir() if p!=meta}))
    emit('finetune_fit',task=task,partition=stage,epochs=epochs,seconds=round(time.time()-started))
    return q


def train(task):
    d=load_task(task)
    prefix=cache_prefix(task,d)
    extra=np.concatenate([d['desc'],d['maccs']],axis=1).astype(np.float32)
    n,nv=len(d['frames']['train']),len(d['frames']['valid'])
    fold=partition(task,d)
    y=d['frames']['train'].Y.to_numpy(float)
    q=np.empty(n)
    dest=OUT/task
    dest.mkdir(parents=True,exist_ok=True)
    for f in np.unique(fold):
        fit,query=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
        assert not set(d['groups'][fit])&set(d['groups'][query])
        assert not set(d['canonical'][fit])&set(d['canonical'][query])
        q[query]=materialize(task,d,prefix,extra,fit,query,f'fold{f}')
    pd.DataFrame(dict(row=np.arange(n),fold=fold,Y_true=y,Y_pred=q)).to_csv(dest/'finetune_oof.csv',index=False)
    qv=materialize(task,d,prefix,extra,np.arange(n),np.arange(n,n+nv),'valid')
    pd.DataFrame(dict(Y_true=d['frames']['valid'].Y,Y_pred=qv)).to_csv(dest/'finetune_valid.csv',index=False)
    evaluate(task)
    del prefix,d
    gc.collect()


def evaluate(task):
    dest=OUT/task
    y,fold,base=anchor_oof(task)
    vy,vb=anchor_valid(task)
    oof=pd.read_csv(dest/'finetune_oof.csv')
    valid=pd.read_csv(dest/'finetune_valid.csv')
    np.testing.assert_array_equal(oof.row,np.arange(len(y)))
    np.testing.assert_array_equal(oof.fold,fold)
    np.testing.assert_array_equal(oof.Y_true,y)
    np.testing.assert_array_equal(valid.Y_true,vy)
    cand,vc=oof.Y_pred.to_numpy(float),valid.Y_pred.to_numpy(float)
    p=read_json(PROTOCOL)
    metric=metric_for(task)
    grid=[]
    for mode in (['raw'] if metric=='MAE' else p['blend_modes']):
        for w in p['blend_grid']:
            if w==0 and mode=='rank':
                continue
            fg=[]
            for f in np.unique(fold):
                mask=fold==f
                q=combine(base[mask],[cand[mask]],[1-w,w],mode,metric)
                fg.append(gain(metric,y[mask],base[mask],q))
            qv=combine(vb,[vc],[1-w,w],mode,metric)
            vg=gain(metric,vy,vb,qv)
            obj=.6*np.mean(fg)+.4*vg-.025*np.std(fg)-.0002*w
            grid.append(dict(mode=mode,weight=w,objective=float(obj),oof_gain=float(np.mean(fg)),
                             valid_gain=float(vg),fold_gains=list(map(float,fg))))
    best=max(grid,key=lambda s:(s['objective'],-s['weight']))
    save_json(dest/'selection.json',dict(task=task,metric=metric,selection=best,grid=grid,
               protocol_sha256=sha(PROTOCOL),baseline='v20',baseline_score=50.019727,
               no_individual_metric_veto=True,selection_caveat=p['uncertainty']))
    emit('v21_selection',task=task,**best)


def final(task):
    if read_json(OUT/task/'selection.json')['selection']['weight']==0:
        return
    d=load_task(task)
    prefix=cache_prefix(task,d)
    extra=np.concatenate([d['desc'],d['maccs']],axis=1).astype(np.float32)
    total=len(d['frames']['train'])+len(d['frames']['valid'])
    q=materialize(task,d,prefix,extra,np.arange(total),np.arange(total,len(d['smiles'])),'final')
    pd.DataFrame(dict(Y_pred=q)).to_csv(OUT/task/'finetune_test.csv',index=False)


def smoke(task):
    d=load_task(task)
    prefix=cache_prefix(task,d)
    n=len(d['frames']['train'])
    extra=np.concatenate([d['desc'],d['maccs']],axis=1).astype(np.float32)
    fit=np.arange(min(64,n))
    y=d['frames']['train'].Y.to_numpy(float)[fit]
    net,pre,tf,_,_=train_one(prefix,extra,fit,y,metric_for(task),21271,1)
    q=predict(net,prefix,np.arange(10),pre.transform(extra[:10]),tf)
    trainable=sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert trainable>5000000
    emit('smoke_passed',task=task,trainable=trainable,prediction_min=float(q.min()),prediction_max=float(q.max()))


if __name__=='__main__':
    remote_only()
    RDLogger.DisableLog('rdApp.*')
    torch.backends.cuda.matmul.allow_tf32=True
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['train','evaluate','final','smoke'])
    parser.add_argument('--tasks',nargs='*')
    args=parser.parse_args()
    for task in args.tasks or read_json(PROTOCOL)['tasks']:
        globals()[args.stage](task)
