"""Complementary metric-aware models, including directed bond message passing."""
import copy
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from scipy.stats import rankdata, norm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from xgboost import XGBClassifier, XGBRegressor
from admet_v6_features import dense, graph_batch, make_graph


class Preprocess:
    def fit(self, x):
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.scaler = StandardScaler().fit(self.imputer.fit_transform(x))
        return self

    def transform(self, x):
        return np.clip(self.scaler.transform(self.imputer.transform(x)), -8, 8).astype(np.float32)


class Target:
    def __init__(self, y, metric):
        self.metric = metric
        self.log = metric == 'Spearman' and y.min() >= 0
        self.y = np.log1p(y) if self.log else y.copy()
        self.mean = float(np.mean(self.y)); self.std = max(float(np.std(self.y)), 1e-6)

    def encode(self, y):
        yy = np.log1p(y) if self.log else y
        return (yy-self.mean)/self.std

    def decode(self, p):
        p = p*self.std+self.mean
        return np.expm1(np.clip(p, -20, 20)) if self.log else p


class ResidualMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(.25))
        self.block = nn.Sequential(nn.Linear(256,256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(.2), nn.Linear(256,256))
        self.head = nn.Sequential(nn.SiLU(), nn.Dropout(.15), nn.Linear(256,1))

    def forward(self, x):
        h = self.project(x)
        return self.head(h+self.block(h)).squeeze(-1)


class DirectedMPNN(nn.Module):
    def __init__(self, atom_dim, bond_dim, extra_dim, hidden=160, depth=3):
        super().__init__()
        self.depth = depth
        self.init = nn.Linear(atom_dim+bond_dim, hidden)
        self.message = nn.Linear(hidden,hidden,bias=False)
        self.atom = nn.Linear(atom_dim+hidden,hidden)
        self.extra = nn.Sequential(nn.Linear(extra_dim,96), nn.LayerNorm(96), nn.SiLU())
        self.drop = nn.Dropout(.15)
        self.head = nn.Sequential(nn.Linear(hidden*2+96,192), nn.ReLU(), nn.Dropout(.2), nn.Linear(192,1))

    def forward(self, batch, extra):
        atom,bond,src,dst,rev,group = batch
        h0 = torch.relu(self.init(torch.cat([atom[src],bond],1)))
        h = h0
        for _ in range(self.depth-1):
            incoming = h.new_zeros((len(atom),h.shape[1])).index_add_(0,dst,h)
            h = self.drop(torch.relu(h0+self.message(incoming[src]-h[rev])))
        incoming = h.new_zeros((len(atom),h.shape[1])).index_add_(0,dst,h)
        ah = torch.relu(self.atom(torch.cat([atom,incoming],1)))
        pooled = ah.new_zeros((len(extra),ah.shape[1])).index_add_(0,group,ah)
        count = torch.bincount(group,minlength=len(extra)).clamp_min(1).to(ah.dtype).unsqueeze(1)
        pooled = torch.cat([pooled/count,pooled/torch.sqrt(count),self.extra(extra)],1)
        return self.head(pooled).squeeze(-1)


def train_loss(pred, target, metric, positive_weight):
    if metric in ['AUROC','AUPRC']:
        loss = nn.functional.binary_cross_entropy_with_logits(pred,target,pos_weight=positive_weight)
        pos, neg = pred[target>.5], pred[target<.5]
        if len(pos) and len(neg):
            # Pairwise regularizer supplements BCE; AP remains the selection metric.
            loss = loss + .10*nn.functional.softplus(neg[None,:]-pos[:,None]).mean()
        return loss
    if metric == 'MAE':
        return nn.functional.l1_loss(pred,target)
    loss = nn.functional.smooth_l1_loss(pred,target,beta=.3)
    delta = target[:,None]-target[None,:]
    mask = delta.abs()>.10
    if mask.any():
        diff = pred[:,None]-pred[None,:]
        loss = loss+.15*nn.functional.softplus(-torch.sign(delta[mask])*diff[mask]).mean()
    return loss


def model_x(data, family):
    if family == 'graph':
        return np.concatenate([data['desc'],data['emb']],1)
    return dense(data)


def fit(family, data, ix, y, metric, protocol):
    classification = metric in ['AUROC','AUPRC']
    bundle = dict(family=family,metric=metric,classification=classification,models=[])
    tf = None if classification else Target(y,metric)
    yy = y.copy() if classification else tf.encode(y)
    bundle['target'] = tf
    if family.startswith('kernel'):
        c = float(family.split('_')[-1])
        model = (SVC(kernel='precomputed',C=c,class_weight='balanced',cache_size=1024)
                 if classification else SVR(kernel='precomputed',C=c,epsilon=.05,cache_size=1024))
        model.fit(data['kernel'][np.ix_(ix,ix)].astype(np.float64),yy)
        bundle.update(models=[model],reference=np.asarray(ix))
        return bundle
    x = model_x(data,family)
    pre = Preprocess().fit(x[ix]); bundle['pre'] = pre
    xx = pre.transform(x[ix])
    if family in ['extra','xgb']:
        for seed in protocol['seeds']:
            if family=='extra':
                cls = ExtraTreesClassifier if classification else ExtraTreesRegressor
                extra = {'class_weight':'balanced'} if classification else {}
                model = cls(n_estimators=450,max_features=.6,min_samples_leaf=2,n_jobs=6,random_state=seed,**extra)
            else:
                cls = XGBClassifier if classification else XGBRegressor
                extra = {'objective':'binary:logistic','scale_pos_weight':float(np.sqrt((1-y.mean())/max(y.mean(),.01)))} if classification else {'objective':'reg:absoluteerror' if metric=='MAE' else 'reg:squarederror'}
                model = cls(n_estimators=850,max_depth=5,learning_rate=.025,min_child_weight=3,
                            subsample=.85,colsample_bytree=.75,reg_lambda=8,reg_alpha=.1,
                            n_jobs=6,tree_method='hist',random_state=seed,**extra)
            model.fit(xx,yy);bundle['models'].append(model)
        return bundle
    if family == 'graph' and 'graphs' not in data:
        data['graphs'] = [make_graph(s) for s in data['smiles']]
    epochs = protocol['graph_epochs'] if family=='graph' else protocol['dense_epochs']
    bundle.update(dim=xx.shape[1],epochs=epochs)
    if family=='graph':
        bundle.update(atom_dim=data['graphs'][0][0].shape[1],bond_dim=14)
    target = torch.as_tensor(yy,device='cuda',dtype=torch.float32)
    xx = torch.as_tensor(xx,device='cuda')
    pw = torch.tensor(min(3.,max(1.,np.sqrt((1-y.mean())/max(y.mean(),.01))))) if classification else torch.tensor(1.)
    pw = pw.cuda()
    for seed in protocol['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model = new_net(bundle).cuda()
        optimizer = torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=2e-5)
        batchsize=64 if family=='graph' else 128
        for epoch in range(epochs):
            model.train();order=rng.permutation(len(ix))
            for start in range(0,len(ix),batchsize):
                jj=order[start:start+batchsize]
                if family=='graph':
                    bat=graph_batch([data['graphs'][ix[j]] for j in jj])
                    prediction=model(bat,xx[jj])
                else:
                    prediction=model(xx[jj])
                loss=train_loss(prediction,target[jj],metric,pw)
                if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),3.);optimizer.step()
            scheduler.step()
            if epoch+1 in [epochs-5,epochs]:
                bundle['models'].append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        del model
    return bundle


def new_net(bundle):
    return (DirectedMPNN(bundle['atom_dim'],bundle['bond_dim'],bundle['dim'])
            if bundle['family']=='graph' else ResidualMLP(bundle['dim']))


def predict(bundle,data,ix):
    family=bundle['family'];preds=[]
    if family.startswith('kernel'):
        x=data['kernel'][np.ix_(ix,bundle['reference'])].astype(np.float64)
    else:
        x=bundle['pre'].transform(model_x(data,family)[ix])
    if family=='graph' and 'graphs' not in data:
        data['graphs']=[make_graph(s) for s in data['smiles']]
    for model in bundle['models']:
        if family.startswith('kernel'):
            p=expit(model.decision_function(x)) if bundle['classification'] else model.predict(x)
        elif family in ['extra','xgb']:
            p=model.predict_proba(x)[:,1] if bundle['classification'] else model.predict(x)
        else:
            net=new_net(bundle).cuda();net.load_state_dict(model);net.eval();parts=[]
            with torch.inference_mode():
                for start in range(0,len(ix),128):
                    xt=torch.as_tensor(x[start:start+128],device='cuda')
                    if family=='graph':
                        pp=net(graph_batch([data['graphs'][i] for i in ix[start:start+128]]),xt)
                    else:pp=net(xt)
                    parts.append(pp.cpu().numpy())
            p=np.concatenate(parts)
            if bundle['classification']:p=expit(p)
            del net
        if not bundle['classification']:p=bundle['target'].decode(p)
        preds.append(p)
    return np.mean(preds,axis=0)
