"""Index earlier local regression recipes too; inspect source, never import it."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    paths=sorted(set(list(ROOT.glob('*.py'))+list((ROOT/'tools').glob('*.py'))))
    records=[]
    for path in paths:
        raw=path.read_bytes()
        try:
            source=raw.decode('utf-8-sig')
            tree=ast.parse(source)
        except (SyntaxError,UnicodeError):continue
        doc=ast.get_docstring(tree) or ''
        names={'METHOD','CONFIG','CONFIGS','TASKS','P','BASES','SEEDS','STEPS','HORIZONS','EPOCHS','BUDGET'}
        definitions=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets):
                definitions.append(ast.get_source_segment(source,node)[:6000])
        # This is a searchable source inventory, not proof every candidate trained.
        if not any(word in (doc+' '.join(definitions)).lower() for word in ('mae','median','quantile','regress','spearman','clearance')):
            continue
        records.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),
                            doc=doc,declarations=definitions))
    value=dict(status='COMPLETE_SOURCE_INDEX',files=len(records),records=records,
               source_scope='Root and tools Python declarations; earlier version recipes included. Not every file is a distinct trained algorithm.',
               no_import_or_fitting=True,limitations='Indexing declarations is not an independent scientific evaluation. Follow result records for completed runs and provenance.')
    (ROOT/'docs/v379_expanded_local_method_index.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Specific early code and existing reviews actually inspected this round.
    inspected=['docs/V145_RESULTS_AND_CODE_GUIDE.md','admet_v23_remote_mtl.py','admet_v24_protocol.json',
               'admet_v44_protocol.json','admet_v52_neural.py','admet_v56_remote_verify.py','docs/history/v56_report.md',
               'tools/v95_graph_specialists.py','tools/v263_fit_label_reliability.py','tools/v264_replicate_noise_gp.py',
               'tools/v297_pair_delta_regression.py','tools/v372_crossfit_experts.py','tools/v374_specialists.py',
               'tools/v131_current_controls.py','tools/v125_low_task_baseline.py','tools/v364_graph_transformer.py',
               'tools/v368_graph_transformer.py','tools/v370_graph_transformer.py']
    inspected=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in inspected]
    (ROOT/'docs/v379_focused_material_review.json').write_text(json.dumps(dict(files=inspected,
        findings=[
            'Directed bond networks, pair differences, duplicate weighting, physical priors, graph transfer, local residuals and nested stacks already have local implementations; do not rename them as unseen innovations.',
            'Several carried legacy regression branches have incomplete checkpoints and complete-ensemble OOF. New honest experts cannot fix missing historical provenance.',
            'v379 tests a narrower capacity/EMA/quantile package plus native-rank objective; no universal improvement or several-point score promise.'
        ]),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(indexed=len(records),focused_read_files=len(inspected),no_fitting=True)))


if __name__=='__main__':main()
