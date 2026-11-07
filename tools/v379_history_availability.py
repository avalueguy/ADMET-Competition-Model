"""Read-only inventory for reconstructing complete historical regression OOF."""
from pathlib import Path
import hashlib
import json
import time


def main():
    root=Path('/root/autodl-tmp/v76')
    records=[]
    for name in ('admet_model_outputs_v23','admet_model_outputs_v44','admet_model_outputs_v46',
                 'admet_model_outputs_v52_neural','admet_model_outputs_v56','admet_model_outputs_v67'):
        folder=root/name
        files=[p for p in folder.rglob('*') if p.is_file()] if folder.exists() else []
        weights=[p for p in files if p.suffix in ('.pt','.cbm','.joblib')]
        predictions=[p for p in files if p.suffix in ('.npy','.npz')]
        records.append(dict(folder=name,exists=folder.exists(),files=len(files),checkpoints=len(weights),
                            arrays=len(predictions),checkpoint_examples=[p.relative_to(root).as_posix() for p in weights[:8]],
                            prediction_examples=[p.relative_to(root).as_posix() for p in predictions[:8]]))
    path=Path('/root/autodl-tmp/regression9_v379/history_availability.json')
    result=dict(at_unix=time.time(),read_only=True,no_fitting=True,records=records,
                limitation='Checkpoint/array existence alone does not establish full historical ensemble provenance or complete OOF. No prediction was recomputed.')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
