"""Stage three: four new endpoint experts on the user-designated CUDA server."""
from pathlib import Path
import hashlib
import json
import numpy as np
import v364_graph_transformer as base

ROOT = Path(__file__).resolve().parent
TASKS = ('ames', 'clearance_microsome_az', 'half_life_obach', 'vdss_lombardo')
base.ROOT = ROOT
base.OUT = ROOT / 'outputs'
base.TASKS = TASKS
base.SEEDS = (3681105, 3681106)
base.FAMILY = {
    'ames': ('dili', 'ld50_zhu', 'herg'),
    'clearance_microsome_az': ('clearance_hepatocyte_az', 'half_life_obach', 'vdss_lombardo'),
    'half_life_obach': ('clearance_hepatocyte_az', 'clearance_microsome_az', 'vdss_lombardo'),
    'vdss_lombardo': ('ppbr_az', 'lipophilicity_astrazeneca', 'half_life_obach'),
}


def signature():
    text = base.sha(__file__) + base.sha(base.__file__) + base.sha(ROOT / 'data_manifest.json') + json.dumps(base.CONFIG, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


base.signature = signature


def configure(net, mode, task):
    if mode == 'scratch':
        return
    for parameter in net.encoder.parameters():
        parameter.requires_grad = False
    if task == 'ames':
        for layer in net.encoder.blocks[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
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


def freeze_protocol():
    value = dict(
        signature=signature(), tasks=TASKS, config=base.CONFIG,
        seeds=base.SEEDS, modes=base.MODES, weights=[0., .2, .45],
        baseline='v364:51.269', training_host='connect.nmb2.seetacloud.com:25870',
        selection='Unique max-min fold0/1 model and weight; freeze before fold2 and valid. No confirmation fallback.',
        confirmation='Strict: all four positive, original practical threshold, and both paired-scaffold bootstrap90 lower bounds positive. Experimental: all four positive.',
        source_sha256=base.sha(__file__), architecture_source_sha256=base.sha(base.__file__),
        external_data=False, external_weights=False, test_labels_used=False,
        self_supervised='Fresh fit-only masked atom/bond and graph-view consistency; exclude union query scaffolds and canonical structures across all four targets.',
        family_transfer=base.FAMILY, auxiliary_budget_steps=120,
        ames='BCE plus pairwise ranking; last two encoder blocks and rank8 adapters.',
        spearman_tasks='Fit-only Gaussian-rank targets, squared error plus pairwise ranking; frozen encoder with rank8 adapters.',
        upstream_controls='v146 cached endpoint controls; partial historical proxy OOF retained. Final controls exactly equal the v364 CSV.',
        public_validation_limitation='Previously used public labels; new endpoint encoders and scaffold bootstrap do not create independent data.',
    )
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
