"""Package current v361 controls and official data for remote Transformer work."""
from pathlib import Path
import csv, hashlib, json, tarfile
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/transformer_stage1_v364'
TASKS=('caco2_wang','clearance_hepatocyte_az','cyp2d6_veith','herg','lipophilicity_astrazeneca','solubility_aqsoldb')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def rank(p):return rankdata(p)/len(p)
def one(pattern):
    ps=list(ROOT.glob(pattern));assert len(ps)==1,(pattern,ps);return ps[0]
def main():
    assert not (OUT/'data_manifest.json').exists(), 'Completed package already exists'
    OUT.mkdir(parents=True,exist_ok=True)
    template=ROOT/'admet_official/submission_template.csv';rows=table(template)
    best=json.loads((ROOT/'docs/current_best_after_v361_20261104.json').read_text(encoding='utf-8'))
    csvfile=ROOT/best['csv'];assert sha(csvfile)==best['csv_sha256']
    current=table(csvfile);tasks=list(dict.fromkeys(r['task'] for r in rows))
    smiles=[];lookup={};mi=[];tid=[];split=[];y=[];evidence={}
    for k,t in enumerate(tasks):
        for si,part in enumerate(('train','valid','test')):
            if si==2:rr=[dict(Drug=r['smiles']) for r in rows if r['task']==t]
            else:
                p=ROOT/'admet_competition_data/data_benchmark'/t/(part+'.csv');rr=table(p);evidence[p.relative_to(ROOT).as_posix()]=sha(p)
            for r in rr:
                sm=r['Drug']
                if sm not in lookup:lookup[sm]=len(smiles);smiles.append(sm)
                mi.append(lookup[sm]);tid.append(k);split.append(si);y.append(float(r['Y']) if si<2 else np.nan)
    chem=ROOT/'admet_model_outputs_v67/features/official_chemistry.npz';meta=json.loads(chem.with_suffix('.json').read_text(encoding='utf-8'))
    assert meta['smiles_sha256']==hashlib.sha256(json.dumps(smiles).encode()).hexdigest()
    z=np.load(chem);groups=z['groups'];canonical=z['canonical']
    np.savez_compressed(OUT/'official_rows.npz',mi=mi,tid=tid,split=split,y=y,smiles=np.array(smiles),groups=groups,canonical=canonical,
                        descriptor=z['x'][:,4096:],fingerprint=z['x'][:,:4096])
    sources=[chem,chem.with_suffix('.json'),template,csvfile]
    records=[]
    for task in TASKS:
        for part in ('fold0','fold1','fold2','valid','final'):
            p=ROOT/f'admet_model_outputs_v146/controls/{task}/{part}.npz';a=np.load(p);fit,q=a['fit'],a['query'];pred=a['p'].copy();sources.append(p)
            if task=='lipophilicity_astrazeneca' and part!='final':
                p=one(f'admet_model_outputs_v304/models/{task}/{part}/*/predictions.npz');extra=np.load(p);np.testing.assert_array_equal(extra['query'],q);pred=.8*pred+.2*extra['p'];sources.append(p)
            if task=='solubility_aqsoldb' and part!='final':
                p=one(f'admet_model_outputs_v329/models/{task}/{part}/*/predictions.npz');extra=np.load(p);np.testing.assert_array_equal(extra['query'],q);pred=.8*pred+.2*extra['p'];sources.append(p)
            if task=='herg' and part!='final':
                experts=[]
                for v in (168,233,333):
                    key={168:'strength1.0',233:'cross_scaffold_alignment',333:'chemical_domains8_groupdro'}[v]
                    p=one(f'admet_model_outputs_v{v}/models/herg/{part}/{key}/predictions.npz');extra=np.load(p);np.testing.assert_array_equal(extra['query'],q);experts.append(rank(extra['p']));sources.append(p)
                pred=.6*rank(pred)+.4*np.mean(experts,axis=0)
            if part=='final':pred=np.array([float(r['y_pred']) for r in current if r['task']==task])
            assert np.isfinite(pred).all() and len(pred)==len(q)
            assert np.all(np.array(split)[fit]<2) and not set(fit)&set(q)
            if part!='final':assert not set(groups[np.array(mi)[fit]])&set(groups[np.array(mi)[q]])
            dest=OUT/'controls'/task/(part+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,fit=fit,query=q,p=pred)
            records.append(dict(task=task,part=part,fit=len(fit),query=len(q),file=dest.relative_to(OUT).as_posix(),sha256=sha(dest),upstream_OOF='partial historical proxy; exact adopted incremental layers replayed'))
    np.save(OUT/'legacy_folds.npy',np.load(ROOT/'recovered_history/legacy_v24_folds.npy'))
    for task in tasks:
        save(OUT/'tasks'/task/'project.json',dict(task=task,status='PILOT' if task in TASKS else 'BASELINE_ONLY_THIS_STAGE',baseline=best,independent_selection=True))
    import shutil
    shutil.copyfile(template,OUT/'submission_template.csv');shutil.copyfile(csvfile,OUT/'v361_baseline.csv')
    save(OUT/'data_manifest.json',dict(tasks=tasks,metrics=[next(r['metric'] for r in rows if r['task']==t) for t in tasks],pilot_tasks=TASKS,
         baseline=best,rows=len(mi),unique_exact_smiles=len(smiles),source_inputs=evidence,
         sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(set(sources))],controls=records,
         test_labels_parsed=False,all_training_remote_only=True,host='connect.nmb2.seetacloud.com',port=24365))
    archive=ROOT/'releases/v364_stage1_inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():tf.add(p,arcname=p.relative_to(OUT).as_posix())
    save(ROOT/'docs/v364_input_package.json',dict(status='PASS',archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,tasks=TASKS,controls=len(records)))
    print(json.dumps(dict(status='PASS',archive=str(archive),bytes=archive.stat().st_size,controls=len(records))))
if __name__=='__main__':main()
