"""Compare the scored v6 payload to v7 without rewriting model predictions."""
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

root=Path.cwd()
oldfile=next(Path('admet_submissions/v6').glob('*.csv'))
newfile=next(Path('admet_submissions/v7').glob('*.csv'))
old=oldfile.read_bytes();new=newfile.read_bytes()
assert hashlib.sha256(old).hexdigest()=='8a677b09bbc8f263ef9ae27055cd2cc88de56a0ed7c7991de67a5256cdb628a4'
assert hashlib.sha256(new).hexdigest()=='88e81b9d16e48cb941f18c5fd6c4a49e067c73659de808e896e51dea8de5acf0'

def read(blob):return list(csv.reader(io.StringIO(blob.decode('utf-8-sig'),newline='')))
def describe(file,blob):
    rows=read(blob);split=1048576
    return dict(file=str(file),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),md5=hashlib.md5(blob).hexdigest(),
                rows=len(rows),data_rows=len(rows)-1,columns=sorted(set(map(len,rows))),
                header_positions=[i+1 for i,r in enumerate(rows) if r and r[0]=='task'],
                utf8_bom=blob.startswith(b'\xef\xbb\xbf'),crlf_count=blob.count(b'\r\n'),lf_count=blob.count(b'\n'),
                ends_crlf=blob.endswith(b'\r\n'),empty_records=sum(not r for r in rows),
                tasks=dict(Counter(r[0] for r in rows[1:])),
                blank_drug_ids=sum(r[2]=='' for r in rows[1:]),
                chunk_count=math.ceil(len(blob)/split),chunk_lengths=[len(blob[i:i+split]) for i in range(0,len(blob),split)],
                second_chunk_begins_on_physical_line=blob[:split].count(b'\n')+1,
                chunk_boundary=repr(blob[split-48:split+48]))

a=read(old);b=read(new)
assert a[0]==b[0] and len(a)==len(b)
changed=Counter();fielddiff=Counter()
for x,y in zip(a[1:],b[1:]):
    assert len(x)==len(y)==6
    for j in range(6):
        if x[j]!=y[j]:fielddiff[a[0][j]]+=1
    assert x[:4]+x[5:]==y[:4]+y[5:]
    if x[4]!=y[4]:changed[x[0]]+=1

# Transplant only the prediction field as raw bytes into the successful envelope.
# rsplit handles commas in quoted SMILES/IDs because the final two fields are plain.
oldlines=old.splitlines(keepends=True);newlines=new.splitlines(keepends=True)
assert len(oldlines)==len(newlines)==16380
rebuilt=[oldlines[0]]
for x,y in zip(oldlines[1:],newlines[1:]):
    ox,op,om=x.rsplit(b',',2);nx,npred,nm=y.rsplit(b',',2)
    assert ox==nx and om==nm
    rebuilt.append(ox+b','+npred+b','+om)
assert b''.join(rebuilt)==new

# A conditional reproduction, not a claim about the unseen server implementation.
whole_plus_chunks=new+new[:1048576]+new[1048576:]
concatenated=read(whole_plus_chunks)
first_unexpected=next((i+1 for i,r in enumerate(concatenated[1:],start=1) if r[0].lstrip('\ufeff') not in set(x[0] for x in b[1:])),None)
assert first_unexpected==16381
report=dict(v6=describe(oldfile,old),v7=describe(newfile,new),field_differences=dict(fielddiff),changed_prediction_rows=dict(changed),
            old_envelope_with_v7_predictions_equals_original_v7_bytes=True,
            conditional_full_file_plus_chunks_first_unknown_task_line=first_unexpected,
            conditional_reproduction_note='Only reproduces the symptom under concatenation. Actual server bytes and logs are unavailable.',
            conclusion='v6 and v7 use exactly the same serialization and metadata. Both exceed current 1 MiB chunk threshold. File size alone cannot explain why v6 succeeded.')
Path('admet_diagnostics/v6_v7_upload_comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(report,indent=2,ensure_ascii=True))
