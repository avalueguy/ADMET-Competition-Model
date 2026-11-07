"""Stage four: independent absorption and CYP-substrate experts, official-only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import v364_graph_transformer as base

ROOT = Path(__file__).resolve().parent
TASKS = ('bioavailability_ma', 'cyp2c9_substrate_carbonmangels',
         'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels')
base.ROOT = ROOT
base.OUT = ROOT / 'outputs'
base.TASKS = TASKS
base.SEEDS = (3691105, 3691106)
base.FAMILY = {
    'bioavailability_ma': ('hia_hou', 'caco2_wang', 'pgp_broccatelli'),
    'cyp2c9_substrate_carbonmangels': ('cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels'),
    'cyp2d6_substrate_carbonmangels': ('cyp2d6_veith', 'cyp2c9_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels'),
    'cyp3a4_substrate_carbonmangels': ('cyp3a4_veith', 'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels'),
}


def signature():
    return hashlib.sha256((base.sha(__file__) + base.sha(base.__file__) +
        base.sha(ROOT / 'data_manifest.json') + json.dumps(base.CONFIG, sort_keys=True)).encode()).hexdigest()


base.signature = signature


def configure(net, mode, task):
    if mode == 'scratch':
        return
    for parameter in net.encoder.parameters():
        parameter.requires_grad = False
    for layer in net.encoder.blocks:
        for parameter in layer.adapter.parameters():
            parameter.requires_grad = True
    for parameter in net.encoder.norm.parameters():
        parameter.requires_grad = True


base.configure = configure
original_global_fit = base.global_fit


def global_fit(d, part):
    fit = original_global_fit(d, part)
    if part != 'final':
        query = np.unique(np.concatenate([base.ctrl(task, part)['query'] for task in TASKS]))
        canonical = d['canonical'][d['mi']]
        assert not np.intersect1d(canonical[fit], canonical[query]).size
    return fit


base.global_fit = global_fit
original_prepare_graphs = base.prepare_graphs


def prepare_graphs(d):
    source = Path('/root/autodl-tmp/transformer_stage3_v368')
    if not (ROOT / 'graphs.pt').exists() and (source / 'graphs.pt').exists():
        assert base.sha(source / 'official_rows.npz') == base.sha(ROOT / 'official_rows.npz')
        cache = torch.load(source / 'graphs.pt', weights_only=False, map_location='cpu')
        expected = hashlib.sha256((base.sha(base.__file__) + str(base.rdBase.rdkitVersion) + str(len(d['smiles']))).encode()).hexdigest()
        assert cache['key'] == expected and len(cache['graphs']) == len(d['smiles'])
        torch.save(cache, ROOT / 'graphs.pt')
        base.save(ROOT / 'graph_audit.json', dict(status='PASS', deterministic_structure_cache_source=str(source),
            source_sha256=base.sha(source / 'graphs.pt'), fresh_encoders_required=True, no_truncation=True,
            maximum_atoms=max(graph['n'] for graph in cache['graphs'])))
    return original_prepare_graphs(d)


base.prepare_graphs = prepare_graphs


def freeze_protocol():
    value = dict(signature=signature(), tasks=TASKS, config=base.CONFIG, seeds=base.SEEDS,
        modes=base.MODES, weights=[0., .2, .45], baseline='v364:51.269',
        training_host='connect.nmb2.seetacloud.com:25870',
        selection='Unique max-min fold0/1 mode and weight, frozen before fold2/valid. No fallback.',
        confirmation='Original strict thresholds; experimental requires all4positive. Retain failed candidates.',
        source_sha256=base.sha(__file__), architecture_source_sha256=base.sha(base.__file__),
        external_data=False, external_weights=False, test_labels_used=False,
        self_supervised='Fresh fit-only masked atom/bond and graph-view consistency, all target query scaffolds and canonical identities excluded.',
        family_transfer=base.FAMILY, auxiliary_budget_steps=120,
        small_tasks='Frozen encoder, rank8 adapters, normalized descriptor branch, BCE plus pairwise ranking.',
        upstream_controls='v146 endpoint caches; partial historical proxy OOF retained. Final predictions exact v364.',
        historical_validation_reuse=True)
    path = base.OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(value))
    else:
        base.save(path, value)


base.freeze_protocol = freeze_protocol


def __getattr__(name):
    return getattr(base, name)


if __name__ == '__main__':
    base.main()
