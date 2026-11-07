"""Stage-two endpoint experiments; reuse audited architecture, refit fold-safe encoders."""
from pathlib import Path
import hashlib, json
import numpy as np
import torch
import v364_graph_transformer as base

ROOT=Path(__file__).resolve().parent
TASKS=('ppbr_az','ld50_zhu')
base.ROOT=ROOT;base.OUT=ROOT/'outputs';base.TASKS=TASKS
base.SEEDS=(3661105,3661106)
base.FAMILY={
 'ppbr_az':('lipophilicity_astrazeneca','solubility_aqsoldb','vdss_lombardo'),
 'ld50_zhu':('ames','dili','solubility_aqsoldb'),
}
def signature():
 return hashlib.sha256((base.sha(__file__)+base.sha(base.__file__)+base.sha(ROOT/'data_manifest.json')+json.dumps(base.CONFIG,sort_keys=True)).encode()).hexdigest()
base.signature=signature

def prepare_graphs(d):
 # These cached objects contain only deterministic atom/bond structure features.
 source=Path('/root/autodl-tmp/transformer_stage1_v364')
 old=json.loads((source/'data_manifest.json').read_text());assert old['unique_exact_smiles']==len(d['smiles'])
 assert base.sha(source/'official_rows.npz')==base.sha(ROOT/'official_rows.npz')
 cache=torch.load(source/'graphs.pt',weights_only=False,map_location='cpu')
 key=hashlib.sha256((base.sha(base.__file__)+str(base.rdBase.rdkitVersion)+str(len(d['smiles']))).encode()).hexdigest()
 assert cache['key']==key and len(cache['graphs'])==len(d['smiles'])
 if not (ROOT/'graph_audit.json').exists():base.save(ROOT/'graph_audit.json',dict(status='PASS',structure_cache_source=str(source/'graphs.pt'),structure_cache_sha256=base.sha(source/'graphs.pt'),no_pretrained_encoder_reused=True,no_truncation=True,maximum_atoms=max(g['n'] for g in cache['graphs'])))
 return cache['graphs']
base.prepare_graphs=prepare_graphs

def configure(net,mode,task):
 if mode=='scratch':return
 for p in net.encoder.parameters():p.requires_grad=False
 if task=='ld50_zhu':
  for layer in net.encoder.blocks[-2:]:
   for p in layer.parameters():p.requires_grad=True
 for layer in net.encoder.blocks:
  for p in layer.adapter.parameters():p.requires_grad=True
 for p in net.encoder.norm.parameters():p.requires_grad=True
base.configure=configure

original_predict=base.predict
def predict(d,graphs,task,part,mode):
 p=original_predict(d,graphs,task,part,mode)
 if task=='ppbr_az':
  p=np.clip(p,0,100);file=base.OUT/'tasks'/task/part/mode/'predictions.npz';z=np.load(file);items={k:z[k] for k in z.files};items['p']=p;z.close()
  np.savez_compressed(file,**items)
  meta=file.with_name('metadata.json');record=json.loads(meta.read_text());record.update(fixed_output_bounds=[0,100],predictions_sha256=base.sha(file));base.save(meta,record)
 return p
base.predict=predict

def freeze_protocol():
 p=base.OUT/'protocol.json';x=dict(signature=signature(),tasks=TASKS,config=base.CONFIG,seeds=base.SEEDS,modes=base.MODES,weights=[0.,.2,.45],baseline='v364:51.269',
  selection='Unique max-min fold0/1 config and weight frozen before confirmation; retain strict thresholds and no fallback',
  confirmation='four positive gains; mean practical threshold and both500 paired-scaffold bootstrap90 lower bounds positive for strict; all4positive only is experimental',
  source_sha256=base.sha(__file__),architecture_source_sha256=base.sha(base.__file__),external_data=False,external_weights=False,test_labels_used=False,training_host='connect.nmb2.seetacloud.com:24365',
  self_supervised='masked atom/bond plus stochastic graph-view consistency, freshly refit after excluding both tasks query scaffolds',
  family_transfer=base.FAMILY,ppbr_head='frozen backbone plus rank8 adapters, L1 target; fixed final clipping0to100',ld50_head='last2layers plus rank8adapters, original-scale L1 without log transform',
  auxiliary_budget_steps=120,upstream_controls='PPBR v30445% layer exact, earlier OOF partially proxy')
 if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(x))
 else:base.save(p,x)
base.freeze_protocol=freeze_protocol

def __getattr__(name):return getattr(base,name)
if __name__=='__main__':base.main()
