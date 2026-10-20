"""Frozen masked-SMILES encoder, pinned weights and resumable feature cache.

No assay targets enter this encoder. Invalid SMILES use a zero embedding;
long SMILES are truncated at 512 tokens and counted in the provenance file.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from transformers import AutoModel, AutoTokenizer

RDLogger.DisableLog('rdApp.*')
MODEL = Path('admet_pretrained/chemberta_mlm')
OUT = Path('admet_pretrained/embeddings')


def canonical(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None else None


class Encoder:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.model = AutoModel.from_pretrained(MODEL, local_files_only=True,
                                              add_pooling_layer=False).to(self.device).eval()

    def encode(self, smiles, batch=64):
        output = np.zeros((len(smiles), 768), dtype=np.float32)
        valid = [(i, canonical(s)) for i, s in enumerate(smiles)]
        valid = [(i, s) for i, s in valid if s is not None]
        valid.sort(key=lambda row: len(row[1]))
        trunc = 0
        with torch.inference_mode():
            for start in range(0, len(valid), batch):
                items = valid[start:start+batch]
                texts = [v for _, v in items]
                tokens = self.tokenizer(texts, padding=True, truncation=True,
                                        max_length=512, return_tensors='pt')
                trunc += sum(len(t) > 512 for t in self.tokenizer(texts, truncation=False)['input_ids'])
                inputs = {k: v.to(self.device) for k, v in tokens.items()}
                h = self.model(**inputs).last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1)
                mean = (h*mask).sum(1)/mask.sum(1).clamp(min=1)
                pooled = torch.cat([mean, h[:, 0]], dim=1).float().cpu().numpy()
                output[[i for i, _ in items]] = pooled
        return output, trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=None)
    ap.add_argument('--tasks', nargs='*')
    args = ap.parse_args()
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    encoder = Encoder(args.device)
    root = Path('admet_competition_data/data_benchmark')
    for task in args.tasks or sorted(p.name for p in root.iterdir()):
        frames = [pd.read_csv(root/task/(s+'.csv')) for s in ['train','valid','test']]
        smiles = pd.concat([d.Drug for d in frames]).astype(str).tolist()
        digest = hashlib.sha256(json.dumps(smiles).encode()).hexdigest()
        dest = OUT/(task+'.npz')
        if dest.exists():
            with np.load(dest) as z:
                if str(z['digest']) == digest:
                    continue
        start = time.time()
        emb, truncated = encoder.encode(smiles)
        assert np.isfinite(emb).all()
        np.savez_compressed(dest, x=emb, digest=digest,
                            n_train=len(frames[0]), n_valid=len(frames[1]))
        info = dict(task=task, rows=len(smiles), truncated=truncated,
                    invalid=sum(canonical(s) is None for s in smiles),
                    model_provenance=str(MODEL/'provenance.json'),
                    feature_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                    elapsed_s=time.time()-start, device=encoder.device)
        dest.with_suffix('.json').write_text(json.dumps(info, indent=2), encoding='utf8')
        print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
