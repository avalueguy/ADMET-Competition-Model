"""Build stage-five orchestration from completed, audited stage utilities."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'


def main():
    for suffix in ('run_stage4', 'verify_remote', 'collect_remote', 'bootstrap'):
        source = TOOLS / f'v369_{suffix}.py'
        target = TOOLS / f"v370_{suffix.replace('stage4', 'stage5')}.py"
        value = source.read_text(encoding='utf-8').replace('v369', 'v370').replace('stage4', 'stage5')
        value = value.replace('stage four', 'stage five').replace('stage-four', 'stage-five')
        target.write_text(value, encoding='utf-8')
    # MAE submission and audit use arithmetic blending, never rank blending.
    for suffix in ('deliver_stage2', 'verify_local'):
        value = (TOOLS / f'v366_{suffix}.py').read_text(encoding='utf-8')
        value = value.replace('v366', 'v370').replace('V366', 'V370').replace('stage2', 'stage5').replace('STAGE2', 'STAGE5')
        value = value.replace('第二阶段 PPBR 与 LD50 实验结果', '第五阶段 图与分子指纹融合、SAM稳健训练结果').replace('第二阶段', '第五阶段')
        value = value.replace('复用已核验的6层图Transformer架构，重新训练排除查询骨架的编码器。比较从零训练、自监督、相关任务迁移，双固定种子。',
                            '五项MAE任务独立建模：新训练6层图Transformer，联合4096维count指纹和分子描述符，比较AdamW与rho=.05的SAM，双固定种子。')
        value = value.replace('PPBR冻结主干并训练rank8适配器，输出固定限制[0,100]；LD50解冻末两层与适配器。两项均优化原尺度L1。',
                            'PPBR冻结主干并训练rank8适配器，输出固定限制[0,100]；其余四项解冻末两层与适配器。全部优化原尺度L1。指纹只做拟合集RMS缩放，不额外取对数。')
        if suffix == 'verify_local':
            value = value.replace("print(json.dumps(dict(status='PASS', no_candidate=True)))",
                "progress = read(ROOT / 'docs/v370_stage5_progress.json')\n        progress.update(status='COMPLETE_NO_ELIGIBLE_CANDIDATE', new_submission_delivered=False, official_score=None, known_best_unchanged='v364:51.269')\n        save(ROOT / 'docs/v370_stage5_progress.json', progress)\n        print(json.dumps(dict(status='PASS', no_candidate=True)))")
        target = TOOLS / f"v370_{suffix.replace('stage2', 'stage5')}.py"
        target.write_text(value, encoding='utf-8')
    # Keep the original architecture source immutable; smoke exercises the actual new input width.
    base = (TOOLS / 'v364_graph_transformer.py').read_text(encoding='utf-8')
    smoke = base[base.index('def smoke(d,graphs):'):base.index('def freeze_protocol():')]
    smoke = smoke.replace("d['descriptor'].shape[1]*2", "(d['descriptor'].shape[1]*2+4096)")
    # Function globals are the wrapper namespace, with other symbols resolved to the original module.
    smoke = smoke.replace('def smoke(d,graphs):', 'def smoke(d,graphs):')
    smoke = smoke.replace(" save(OUT/'smoke.json'", " sam_before=bench.head[-1].weight.detach().clone();bench.train();before_all={n:p.detach().clone() for n,p in bench.named_parameters()}\n sam_loss,robust_loss=sam_update(bench,opt,b,torch.zeros((64,(d['descriptor'].shape[1]*2+4096)),device='cuda'),torch.zeros(64,device='cuda'),'MAE')\n assert not torch.equal(sam_before,bench.head[-1].weight);assert all(torch.isfinite(p).all() for p in bench.parameters())\n fp_start=bench.desc.descriptor_dim;assert bench.desc.fingerprint[0].weight.grad is not None and bench.desc.numeric[0].weight.grad is not None\n save(OUT/'smoke.json'")
    smoke = smoke.replace('actual_gradient_update=True,', 'actual_gradient_update=True,sam_gradient_update=True,sam_parameter_restore_exact=True,fp_descriptor_branch_gradients=True,')
    wrapper = TOOLS / 'v370_graph_transformer.py'
    value = wrapper.read_text(encoding='utf-8')
    assert 'def smoke(d,graphs):' not in value, 'One-time generator: preserve completed model source'
    # Materialize base symbol bindings once to preserve conventional Python name resolution.
    value += '\n\n' + '\n'.join(f'{name} = base.{name}' for name in
        ('setup', 'Pretrain', 'batch', 'objective', 'global_fit', 'ctrl', 'save', 'OUT', 'emit')) + '\n\n'
    value += smoke + "\nbase.smoke = smoke\n\nif __name__ == '__main__':\n    base.main()\n"
    wrapper.write_text(value, encoding='utf-8')
    progress = json.loads((ROOT / 'docs/v369_stage4_progress.json').read_text(encoding='utf-8'))
    progress.update(status='COMPLETE_NO_ELIGIBLE_CANDIDATE', new_submission_delivered=False,
                    confirmed_failed_task='bioavailability_ma', known_best_unchanged='v364:51.269')
    (ROOT / 'docs/v369_stage4_progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    progress = dict(status='PREPARING', version='v370', baseline='v364:51.269',
        tasks=['caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb'],
        modes=['selfsupervised', 'sam'], seeds=[3701105,3701106], weights=[0.,.2,.45],
        training_host='connect.nmb2.seetacloud.com:25870', external_data=False, external_weights=False,
        independent_fold_encoders=True, confirmation_fallback=False, official_score=None,
        experiment='Graph + count fingerprint, AdamW versus fixed-rho SAM; final bounded experiment this session.')
    (ROOT / 'docs/v370_stage5_progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
