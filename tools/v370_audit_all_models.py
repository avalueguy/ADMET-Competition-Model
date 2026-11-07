"""Replay every completed expert, including rejected development candidates."""
from pathlib import Path
import json
import numpy as np
import torch
import v370_graph_transformer as m
from v370_verify_remote import manual_encoder

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    data, signature = m.load(), m.signature()
    graphs = m.prepare_graphs(data)
    files = sorted((m.OUT / 'tasks').glob('*/*/*/predictions.npz'))
    assert len(files) >= 20
    records = []
    for path in files:
        task, part, mode = path.parts[-4:-1]
        control = m.ctrl(task, part)
        fit, query = control['fit'], control['query']
        with np.load(path) as z:
            saved = {k:z[k] for k in z.files}
        assert str(saved['signature']) == signature
        np.testing.assert_array_equal(saved['fit'], fit)
        np.testing.assert_array_equal(saved['query'], query)
        assert not np.intersect1d(fit, query).size
        if part != 'final':
            assert not np.intersect1d(data['rg'][fit], data['rg'][query]).size
            assert not np.intersect1d(data['canonical'][data['mi'][fit]], data['canonical'][data['mi'][query]]).size
        desc, ds = m.normalize_desc(data, fit)
        yy, ys = m.target(data['y'][fit], 'MAE')
        changed = {**data, 'y':data['y'].copy()}
        changed['y'][query] = np.arange(len(query)) + 7654321.
        second, second_stats = m.normalize_desc(changed, fit)
        np.testing.assert_array_equal(desc, second)
        transformed, second_target = m.target(changed['y'][fit], 'MAE')
        np.testing.assert_array_equal(yy, transformed)
        assert ys == second_target
        sample = query[np.unique(np.linspace(0,len(query)-1,min(8,len(query)),dtype=int))]
        indices = np.searchsorted(query, sample)
        xx = torch.as_tensor(desc[data['mi'][sample]])
        device_desc = torch.as_tensor(desc, device='cuda')
        b = m.batch(graphs, data['mi'][sample], device='cpu')
        checks, members = [], []
        for seed_index, seed in enumerate(m.SEEDS):
            checkpoint = path.parent / f'model_{seed}.pt'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert state['signature'] == signature and state['mode'] == mode
            np.testing.assert_array_equal(state['fit'], fit)
            np.testing.assert_array_equal(state['query'], query)
            for key in ds:
                np.testing.assert_array_equal(state['descriptor'][key], ds[key])
            assert state['target'] == ys
            net = m.Predictor(desc.shape[1]).eval()
            net.load_state_dict(state['weights'])
            with torch.inference_mode():
                expected = net.head(torch.cat((manual_encoder(net.encoder,b), net.desc(xx)),dim=1)).squeeze(1).numpy()
                cpu = net(b, xx).numpy()
            manual_error = float(np.max(np.abs(expected-cpu)))
            cpu_error = float(np.max(np.abs(cpu-saved['raw_members'][seed_index,indices])))
            assert manual_error < .00005 and cpu_error < .0001, (task,part,mode,manual_error,cpu_error)
            raw = m.infer(net.cuda(),data,graphs,query,device_desc)
            replay_error = float(np.max(np.abs(raw-saved['raw_members'][seed_index])))
            assert replay_error < .00001, replay_error
            members.append(raw)
            checks.append(dict(seed=seed,checkpoint_sha256=m.sha(checkpoint),
                manual_attention_error=manual_error,cpu_error=cpu_error,full_replay_error=replay_error))
            pre = torch.load(m.OUT/'encoders'/part/str(seed)/'pretrained.pt',map_location='cpu',weights_only=False)
            permitted = m.global_fit(data, part)
            np.testing.assert_array_equal(pre['fit_rows'],permitted)
            np.testing.assert_array_equal(pre['fit_molecules'],np.unique(data['mi'][permitted]))
            assert pre['signature'] == signature
            del net
        expected = np.mean(members,axis=0)*ys['scale']+ys['center']
        if task == 'ppbr_az':
            expected = np.clip(expected,0,100)
        np.testing.assert_allclose(expected,saved['p'],atol=1e-6,rtol=1e-5)
        records.append(dict(task=task,part=part,mode=mode,query_rows=len(query),
            fit_rows=len(fit),checks=checks,query_label_perturbation_invariant=True))
        del device_desc
        torch.cuda.empty_cache()
        print(json.dumps(dict(stage='audited',task=task,part=part,mode=mode)),flush=True)
    value = dict(status='PASS',signature=signature,source_sha256=m.sha(__file__),expert_jobs=len(records),
        checkpoints=sum(len(r['checks']) for r in records),records=records,
        full_replay_max_error=max(c['full_replay_error'] for r in records for c in r['checks']),
        manual_attention_max_error=max(c['manual_attention_error'] for r in records for c in r['checks']),
        cpu_max_error=max(c['cpu_error'] for r in records for c in r['checks']),
        no_training=True,no_model_selection=True,test_labels_accessed=False)
    m.save(ROOT/'all_models_audit.json',value)
    print(json.dumps({k:v for k,v in value.items() if k!='records'}),flush=True)


if __name__ == '__main__':
    main()
