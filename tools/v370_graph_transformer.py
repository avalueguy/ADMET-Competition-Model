"""Stage five: graph/count-fingerprint fusion, AdamW versus fixed-rho SAM."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from torch import nn
import v364_graph_transformer as base

ROOT = Path(__file__).resolve().parent
TASKS = ('caco2_wang', 'ld50_zhu', 'lipophilicity_astrazeneca', 'ppbr_az', 'solubility_aqsoldb')
base.ROOT = ROOT
base.OUT = ROOT / 'outputs'
base.TASKS = TASKS
base.SEEDS = (3701105, 3701106)
base.MODES = ('selfsupervised', 'sam')
base.CONFIG.update(fingerprint_width=128, fingerprint_rms_floor=.25,
                   fingerprint_clip=8., sam_rho=.05)


def signature():
    return hashlib.sha256((base.sha(__file__) + base.sha(base.__file__) +
        base.sha(ROOT / 'data_manifest.json') + json.dumps(base.CONFIG, sort_keys=True)).encode()).hexdigest()


base.signature = signature
original_normalize = base.normalize_desc


def normalize_desc(d, fit):
    descriptor, stats = original_normalize(d, fit)
    raw = d['fingerprint']
    assert raw.shape[1] == 4096 and np.isfinite(raw).all() and raw.min() >= 0
    selected = raw[np.unique(d['mi'][fit])].astype(np.float64)
    rms = np.sqrt(np.mean(selected ** 2, axis=0)).clip(base.CONFIG['fingerprint_rms_floor'])
    fp = np.clip(raw / rms, 0, base.CONFIG['fingerprint_clip']).astype(np.float32)
    stats['fingerprint_rms'] = rms
    return np.concatenate((descriptor, fp), axis=1), stats


base.normalize_desc = normalize_desc


class Features(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.descriptor_dim = dim - 4096
        assert self.descriptor_dim > 0
        self.numeric = nn.Sequential(nn.Linear(self.descriptor_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(.2))
        self.fingerprint = nn.Sequential(nn.Linear(4096, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(.2),
                                         nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU())

    def forward(self, x):
        return torch.cat((self.numeric(x[:, :self.descriptor_dim]), self.fingerprint(x[:, self.descriptor_dim:])), dim=1)


class Predictor(nn.Module):
    def __init__(self, desc_dim):
        super().__init__()
        self.encoder = base.Encoder()
        self.desc = Features(desc_dim)
        self.head = nn.Sequential(nn.Linear(base.CONFIG['width'] * 2 + 192, 128), nn.GELU(), nn.Dropout(.2), nn.Linear(128, 1))
        self.use_sam = False

    def forward(self, b, desc):
        return self.head(torch.cat((self.encoder(b)[1], self.desc(desc)), dim=1)).squeeze(1)


base.Predictor = Predictor


def configure(net, mode, task):
    net.use_sam = mode == 'sam'
    for p in net.encoder.parameters():
        p.requires_grad = False
    if task != 'ppbr_az':
        for layer in net.encoder.blocks[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
    for layer in net.encoder.blocks:
        for p in layer.adapter.parameters():
            p.requires_grad = True
    for p in net.encoder.norm.parameters():
        p.requires_grad = True


base.configure = configure
ordinary_updates = base.fit_updates


def sam_update(net, opt, b, xx, yy, metric):
    """Two passes on the same batch/dropout realization, then one AdamW step."""
    opt.zero_grad(set_to_none=True)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = base.objective(net(b, xx), yy, metric)
    assert torch.isfinite(loss)
    loss.backward()
    parameters = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
    assert parameters and all(torch.isfinite(p.grad).all() for p in parameters)
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in parameters]))
    offsets = [(p, p.grad.detach().clone() * (base.CONFIG['sam_rho'] / (norm + 1e-12))) for p in parameters]
    originals = [p.detach().clone() for p, _ in offsets]
    with torch.no_grad():
        for p, offset in offsets:
            p.add_(offset)
    opt.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            robust_loss = base.objective(net(b, xx), yy, metric)
        assert torch.isfinite(robust_loss)
        robust_loss.backward()
    finally:
        with torch.no_grad():
            for (p, _), original in zip(offsets, originals):
                p.copy_(original)
    assert all(torch.equal(p, original) for (p, _), original in zip(offsets, originals))
    assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
    opt.step()
    return float(loss.detach()), float(robust_loss.detach())


def fit_updates(net, opt, sch, d, graphs, fit, tx, desc, metric, steps, rng, label):
    if not net.use_sam:
        return ordinary_updates(net, opt, sch, d, graphs, fit, tx, desc, metric, steps, rng, label)
    losses, robust = [], []
    start = time.monotonic()
    for step in range(steps):
        net.train()
        ix = rng.choice(len(fit), base.CONFIG['finetune_batch'], replace=len(fit) < base.CONFIG['finetune_batch'])
        rows = fit[ix]
        b = base.batch(graphs, d['mi'][rows], rng=rng)
        loss, perturbed = sam_update(net, opt, b, desc[d['mi'][rows]], tx[ix], metric)
        sch.step()
        losses.append(loss)
        robust.append(perturbed)
        if (step + 1) % 200 == 0:
            base.emit('sam_progress', label=label, step=step+1, steps=steps, loss=float(np.mean(losses[-100:])), seconds=time.monotonic()-start)
    return dict(seconds=time.monotonic()-start, first100=float(np.mean(losses[:100])),
                last100=float(np.mean(losses[-100:])), robust_last100=float(np.mean(robust[-100:])),
                rho=base.CONFIG['sam_rho'], parameter_restore_exact=True, paired_dropout=True)


base.fit_updates = fit_updates
original_global_fit = base.global_fit


def global_fit(d, part):
    fit = original_global_fit(d, part)
    if part != 'final':
        query = np.unique(np.concatenate([base.ctrl(t, part)['query'] for t in TASKS]))
        identity = d['canonical'][d['mi']]
        assert not np.intersect1d(identity[fit], identity[query]).size
    return fit


base.global_fit = global_fit


def prepare_graphs(d):
    source = Path('/root/autodl-tmp/transformer_stage3_v368')
    assert base.sha(source / 'official_rows.npz') == base.sha(ROOT / 'official_rows.npz')
    cache = torch.load(source / 'graphs.pt', weights_only=False, map_location='cpu')
    key = hashlib.sha256((base.sha(base.__file__) + str(base.rdBase.rdkitVersion) + str(len(d['smiles']))).encode()).hexdigest()
    assert cache['key'] == key and len(cache['graphs']) == len(d['smiles'])
    path = ROOT / 'graph_audit.json'
    if not path.exists():
        base.save(path, dict(status='PASS', structure_cache_source=str(source / 'graphs.pt'),
            structure_cache_sha256=base.sha(source / 'graphs.pt'), no_pretrained_encoder_reused=True,
            no_truncation=True, maximum_atoms=max(g['n'] for g in cache['graphs'])))
    return cache['graphs']


base.prepare_graphs = prepare_graphs
original_predict = base.predict


def predict(d, graphs, task, part, mode):
    p = original_predict(d, graphs, task, part, mode)
    if task == 'ppbr_az':
        p = np.clip(p, 0, 100)
        path = base.OUT / 'tasks' / task / part / mode / 'predictions.npz'
        with np.load(path) as z:
            value = {k: z[k] for k in z.files}
        value['p'] = p
        np.savez_compressed(path, **value)
        meta = path.with_name('metadata.json')
        value = json.loads(meta.read_text())
        value.update(fixed_output_bounds=[0, 100], predictions_sha256=base.sha(path))
        base.save(meta, value)
    return p


base.predict = predict


def freeze_protocol():
    value = dict(signature=signature(), tasks=TASKS, config=base.CONFIG, seeds=base.SEEDS,
        modes=base.MODES, weights=[0., .2, .45], baseline='v364:51.269',
        selection='Unique max-min fold0/1 mode and weight; frozen before fold2/valid; no fallback.',
        confirmation='Strict all4positive, original practical threshold, both paired scaffold bootstrap90 lower bounds positive; experimental all4positive.',
        source_sha256=base.sha(__file__), architecture_source_sha256=base.sha(base.__file__),
        training_host='connect.nmb2.seetacloud.com:25870', external_data=False, external_weights=False, test_labels_used=False,
        architecture='Fresh 6-layer graph Transformer + independent descriptor64 and count-fingerprint128 branches + common prediction head.',
        fingerprint='Existing 4096-column count representation; no additional logarithm/binarization. Fit-only RMS normalization floor.25, fixed clip8.',
        training='Fresh fit-only pretraining per partition and seed. Last2layers+rank8adapters, except PPBR adapters only. Original-scale L1.',
        comparison='Same seed/data/architecture/steps: standard AdamW versus nonadaptive SAM rho.05 with AdamW, same dropout masks for both passes.',
        small_task='PPBR fixed output clipping[0,100]; LD50 no target log transform.',
        upstream_controls='Replay all adopted v364 layers. Partial historical proxy OOF; public labels repeatedly reused.',
        no_learned_encoder_reused_from_previous_stages=True)
    path = base.OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(value))
    else:
        base.save(path, value)


base.freeze_protocol = freeze_protocol


def __getattr__(name):
    return getattr(base, name)



setup = base.setup
Pretrain = base.Pretrain
batch = base.batch
objective = base.objective
global_fit = base.global_fit
ctrl = base.ctrl
save = base.save
OUT = base.OUT
emit = base.emit

def smoke(d,graphs):
 setup();torch.manual_seed(364);rng=np.random.default_rng(364);ids=np.array([1,7,39,200]);net=Pretrain().cuda();before=net.element.weight.detach().clone();opt=torch.optim.AdamW(net.parameters(),lr=.001)
 b=batch(graphs,ids,mask_probability=.2,rng=rng);loss,z=net(b);loss.backward();assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None);opt.step();assert not torch.equal(before,net.element.weight)
 net.eval();torch.backends.cuda.matmul.allow_tf32=False
 with torch.inference_mode():
  aa=net.encoder(batch(graphs,ids))[1];bb=net.encoder(batch(graphs,ids,permutation=True,rng=rng))[1];error=float((aa-bb).abs().max());assert error<.0001,error
  single=net.encoder(batch(graphs,ids[:1]))[1];batch_error=float((single-aa[:1]).abs().max());assert batch_error<.0001,batch_error
 # Check direction and magnitude of the four task losses with known examples.
 loss_checks={}
 for metric in ('MAE','Spearman','AUROC','AUPRC'):
  y=torch.tensor([-1.,0.,1.,2.] if metric in ('MAE','Spearman') else [0.,0.,1.,1.],device='cuda')
  good=y if metric in ('MAE','Spearman') else torch.tensor([-3.,-2.,2.,3.],device='cuda')
  bad=good.flip(0);a=float(objective(good,y,metric));b=float(objective(bad,y,metric));assert a<b,(metric,a,b);loss_checks[metric]=dict(good=a,bad=b)
 # Strict train/query isolation is checked for all four development parts.
 overlap=[]
 for part in ('fold0','fold1','fold2','valid'):
  fit=global_fit(d,part);query=np.unique(np.concatenate([ctrl(t,part)['query'] for t in TASKS]));assert not set(d['rg'][fit])&set(d['rg'][query]);overlap.append(dict(part=part,fit=len(fit),query=len(query),group_overlap=0))
 # Throughput calibration uses structures and synthetic labels, never query outcomes.
 del net,opt;torch.cuda.empty_cache();bench=Predictor((d['descriptor'].shape[1]*2+4096)).cuda();opt=torch.optim.AdamW(bench.parameters(),lr=.0004)
 fit=global_fit(d,'fold0');pool=np.unique(d['mi'][fit]);features,feature_stats=normalize_desc(d,fit);features=torch.as_tensor(features,device='cuda');times=[];torch.cuda.reset_peak_memory_stats()
 for step in range(25):
  ix=rng.choice(pool,64,replace=False);t0=time.monotonic();b=batch(graphs,ix);opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=bench(b,features[ix]);ll=pred.square().mean()
  ll.backward();opt.step();torch.cuda.synchronize();times.append(time.monotonic()-t0)
 timing=dict(median_seconds=float(np.median(times[5:])),p95_seconds=float(np.quantile(times[5:],.95)),peak_gpu_bytes=torch.cuda.max_memory_allocated(),steps=25,batch=64)
 sam_before=bench.head[-1].weight.detach().clone();bench.train()
 sam_loss,robust_loss=sam_update(bench,opt,b,features[ix],torch.zeros(64,device='cuda'),'MAE')
 assert not torch.equal(sam_before,bench.head[-1].weight);assert all(torch.isfinite(p).all() for p in bench.parameters())
 assert bench.desc.fingerprint[0].weight.grad.abs().sum()>0 and bench.desc.numeric[0].weight.grad.abs().sum()>0
 save(OUT/'smoke.json',dict(status='PASS',signature=signature(),cuda_device=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in bench.parameters()),loss=float(loss.detach()),permutation_max_error=error,batch_max_error=batch_error,loss_checks=loss_checks,actual_gradient_update=True,sam_gradient_update=True,sam_parameter_restore_exact=True,fp_descriptor_branch_gradients=True,no_truncation=True,isolation=overlap,benchmark=timing));emit('smoke_pass',permutation_max_error=error,benchmark=timing)

base.smoke = smoke

if __name__ == '__main__':
    base.main()
