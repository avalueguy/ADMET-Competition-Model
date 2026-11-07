"""Prepare two new endpoint experts against the measured v364 baseline."""
from pathlib import Path
import csv, hashlib, json, shutil, tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/transformer_stage2_v366'
TASKS=('ppbr_az','ld50_zhu')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def table(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 assert not (OUT/'data_manifest.json').exists(),'Preserve completed input package'
 OUT.mkdir(parents=True,exist_ok=True)
 previous=ROOT/'experiments/transformer_stage1_v364';old=json.loads((previous/'data_manifest.json').read_text());best=json.loads((ROOT/'docs/current_best_after_v364_20261104.json').read_text())
 baseline=ROOT/best['csv'];assert sha(baseline)==best['csv_sha256'];current=table(baseline)
 raw=previous/'official_rows.npz';shutil.copyfile(raw,OUT/raw.name);z=np.load(raw);assert np.isnan(z['y'][z['split']==2]).all()
 groups=z['groups'][z['mi']];identity=z['canonical'][z['mi']];records=[];sources=[]
 for task in TASKS:
  for part in ('fold0','fold1','fold2','valid','final'):
   src=ROOT/'admet_model_outputs_v146/controls'/task/(part+'.npz');a=np.load(src);fit,q=a['fit'],a['query'];p=a['p'].copy();sources.append(src)
   if task=='ppbr_az' and part!='final':
    candidates=list((ROOT/'admet_model_outputs_v304/models'/task/part).glob('*/predictions.npz'));assert len(candidates)==1,candidates
    expert=np.load(candidates[0]);np.testing.assert_array_equal(expert['query'],q);p=.55*p+.45*expert['p'];sources.append(candidates[0])
   if part=='final':p=np.array([float(r['y_pred']) for r in current if r['task']==task])
   assert np.isfinite(p).all() and np.isfinite(z['y'][fit]).all();assert not np.intersect1d(fit,q).size
   assert np.all(np.diff(q)>0) and np.all(z['tid'][fit]==old['tasks'].index(task)) and np.all(z['tid'][q]==old['tasks'].index(task))
   if part!='final':assert not np.intersect1d(groups[fit],groups[q]).size and not np.intersect1d(identity[fit],identity[q]).size
   dest=OUT/'controls'/task/(part+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,fit=fit,query=q,p=p)
   records.append(dict(task=task,part=part,fit=len(fit),query=len(q),file=dest.relative_to(OUT).as_posix(),sha256=sha(dest),upstream_OOF='Partial historical proxy; adopted PPBR v30445% layer replayed exactly'))
 shutil.copyfile(baseline,OUT/'v364_baseline.csv');shutil.copyfile(previous/'submission_template.csv',OUT/'submission_template.csv')
 value=dict(tasks=old['tasks'],metrics=old['metrics'],pilot_tasks=TASKS,baseline=best,rows=old['rows'],unique_exact_smiles=old['unique_exact_smiles'],
  official_rows_sha256=sha(raw),official_rows_source='transformer_stage1_v364/official_rows.npz',source_inputs=old['source_inputs'],
  sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(set(sources))],controls=records,
  test_labels_parsed=False,all_training_remote_only=True,host='connect.nmb2.seetacloud.com',port=24365)
 save(OUT/'data_manifest.json',value)
 archive=ROOT/'releases/v366_stage2_inputs.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for p in sorted(OUT.rglob('*')):
   if p.is_file() and p.name!='official_rows.npz':tf.add(p,arcname=p.relative_to(OUT).as_posix())
 report=dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,official_rows_sha256=sha(raw),official_rows_to_copy_on_remote=True,controls=records)
 save(ROOT/'docs/v366_input_package.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
