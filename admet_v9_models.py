"""Sample-count-controlled gated training; v8 algorithms remain immutable."""
import json
import math
import numpy as np
import torch
from torch import nn

from admet_v6_models import Preprocess, train_loss
from admet_v8_models import load_task, RobustTarget, GatedAssayNetwork, transfer_training, predict as predict8


def fit(family, d, ix, y, metric, protocol):
    assert family=='gated_short' and torch.cuda.is_available()
    records, audits=transfer_training(d,ix,y,metric)
    pre=[Preprocess().fit(np.concatenate([r[1][k] for r in records])) for k in range(5)]
    xx=[]; yy=[]; transforms=[]; weights=[]; metrics=[]
    for name,vx,labels,m in records:
        cls=m in ['AUROC','AUPRC']; tf=None if cls else RobustTarget(labels,m)
        transforms.append(tf); metrics.append(m)
        xx.append([torch.as_tensor(p.transform(a),device='cuda') for p,a in zip(pre,vx)])
        yy.append(torch.as_tensor(labels if cls else tf.encode(labels),dtype=torch.float32,device='cuda'))
        weight=min(3.,max(1.,np.sqrt((1-labels.mean())/max(labels.mean(),.01)))) if cls else 1.
        weights.append(torch.tensor(weight,dtype=torch.float32,device='cuda'))
    dims=[x.shape[1] for x in xx[0]]
    bundle=dict(family=family,metric=metric,classification=metric in ['AUROC','AUPRC'],target=transforms[0],
        feature_key=d['v8_feature_key'],task=d['task'],models=[],pre=pre,dimensions=dims,
        names=[r[0] for r in records],hidden=protocol['hidden'],source_audit=audits,protocol=protocol,
        target_rows=len(ix),target_passes_per_seed=protocol['epochs'])
    for seed in protocol['seeds']:
        torch.manual_seed(seed); rng=np.random.default_rng(seed)
        net=GatedAssayNetwork(dims,len(records),protocol['hidden']).cuda()
        optimizer=torch.optim.AdamW(net.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,protocol['epochs'],eta_min=2e-5)
        for epoch in range(1,protocol['epochs']+1):
            net.train(); order=rng.permutation(len(ix))
            for start in range(0,len(ix),protocol['target_batch']):
                draws=[order[start:start+protocol['target_batch']]]
                draws += [rng.integers(len(v),size=protocol['aux_batch']) for v in yy[1:]]
                bx=[torch.cat([v[k][idx] for v,idx in zip(xx,draws)]) for k in range(5)]
                bt=torch.cat([torch.full((len(idx),),k,device='cuda',dtype=torch.long) for k,idx in enumerate(draws)])
                pred=net(bx,bt); losses=[]
                for k,idx in enumerate(draws):
                    target=yy[k][idx]
                    if metrics[k] in ['AUROC','AUPRC']:
                        target=(1-protocol['label_smoothing'])*target+protocol['label_smoothing']*.5
                    loss=train_loss(pred[bt==k],target,metrics[k],weights[k])
                    losses.append((2. if k==0 else 1.)*loss)
                loss=torch.stack(losses).sum()/(len(records)+1)
                if not torch.isfinite(loss): raise ValueError('nonfinite v9 loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),2.); optimizer.step()
            scheduler.step()
            if epoch in protocol['snapshots']:
                bundle['models'].append({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        print(json.dumps(dict(stage='v9_seed_done',task=d['task'],seed=seed,epochs=protocol['epochs'],
            target_rows=len(ix),target_presentations=len(ix)*protocol['epochs'],last_loss=float(loss.detach()))),flush=True)
        del net
    return bundle


def predict(bundle,d,ix):
    return predict8(bundle,d,ix)
