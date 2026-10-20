import json
from pathlib import Path

root=Path('/home/czp123/admet_v23_20260928')
out=root/'admet_model_outputs_v46'
files=[]
for path in list(out.glob('*.json'))+list((root/'admet_submissions/v46').glob('*')):
    if path.is_file():files.append(str(path.relative_to(root)))
(out/'collect_files.json').write_text(json.dumps(files,ensure_ascii=False,indent=2))
print(json.dumps(files,ensure_ascii=False,indent=2))
