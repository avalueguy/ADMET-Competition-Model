"""Small real GPU fit and replay, with group-safe neighbors; no outer scoring."""
from pathlib import Path
import json,time
import numpy as np,torch
import v378_retrieval_transformer as m


def main():
 m.setup();d=m.load();a=m.design(d,m.TASKS[0],'fold0');x,qx,transform=m.features(d,a);ns=m.neighbors(d,a)
 for row,indices in zip(a['fit'],ns[0][0]):assert not np.any(d['rg'][row]==d['rg'][a['fit'][indices]])
 ix=ns[1][0][:8];tx=torch.as_tensor(x,device='cuda');y=torch.as_tensor(d['y'][a['fit']],dtype=torch.float32,device='cuda');q=torch.as_tensor(qx[:8],device='cuda')
 for b in m.BASES:
  net=m.Net(transform['extra'],b['retrieval']).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0005);start=time.monotonic()
  for _ in range(12):
   net.train();h=net.encode(tx[:32])
   if b['retrieval']:
    refs=ns[0][0][:32];ref=net.encode(tx[refs.reshape(-1)]).reshape(32,len(refs[0]),64)
    pred=net.combine(h,ref,y[refs],torch.as_tensor(ns[0][1][:32],device='cuda'))
   else:pred=net.direct(h).squeeze(1)
   loss=(pred-y[:32]).abs().mean();opt.zero_grad();loss.backward();opt.step();assert torch.isfinite(loss)
  qn=tuple(torch.as_tensor(v[:len(q)],device='cuda') for v in ns[1]) if b['retrieval'] else None
  p=m.infer(net,tx,q,y,qn);assert np.isfinite(p).all();print(json.dumps(dict(base=b['name'],seconds12=time.monotonic()-start)),flush=True)
 print('SMOKE_PASS')


if __name__=='__main__':main()
