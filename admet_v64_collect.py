"""Copy only lightweight completion evidence and CSVs from the training host."""
import json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
HOST='czp123@100.109.143.42'
REMOTE='/home/czp123/admet_v23_20260928'

def main():
    remote_script="""import json
from pathlib import Path
root=Path('/home/czp123/admet_v23_20260928')
required=root/'admet_model_outputs_v64/pipeline_status.json'
assert required.exists() and json.loads(required.read_text())['status']=='COMPLETE'
files=[]
names=['analysis.json','fresh.json','release.json','no_release.json','fusion_replay.json','model_replay.json','all_completed.json','independent_chemical_gate.json','domain_check.json','pipeline_status.json']
for version in [61,62,63,65,64]:
    for name in names:
        file=root/f'admet_model_outputs_v{version}'/name
        if file.exists():files.append(str(file.relative_to(root)))
    folder=root/f'admet_submissions/v{version}'
    if folder.exists():files.extend(str(file.relative_to(root)) for file in folder.iterdir() if file.suffix in ['.csv','.json'])
print(json.dumps(files))
"""
    result=subprocess.run(['ssh','-o','BatchMode=yes',HOST,'cd '+REMOTE+' && .venv/bin/python -'],input=remote_script,text=True,capture_output=True,check=True)
    files=json.loads(result.stdout)
    for name in files:
        path=ROOT/name
        assert path.resolve().is_relative_to(ROOT.resolve())
        path.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp',HOST+':'+REMOTE+'/'+name,str(path)],check=True)
    print(json.dumps(dict(status='COLLECTED',files=len(files),large_models_downloaded=False)))

if __name__=='__main__':main()
