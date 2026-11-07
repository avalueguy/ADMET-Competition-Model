"""Small real remote fit and saved-state replay; no outer-label evaluation."""
from pathlib import Path
import json
import numpy as np
import torch
import v379_distribution_transformer as m


def main():
    m.setup()
    d=m.load()
    a=m.design(d,m.TASKS[0],'fold0')
    xx,qq,state=m.features(d,a)
    y,_,_,_=m.base.transform(d['y'][a['fit']],False)
    x=torch.as_tensor(xx[:32],device='cuda')
    q=torch.as_tensor(qq[:8],device='cuda')
    y=torch.as_tensor(y[:32],dtype=torch.float32,device='cuda')
    records=[]
    for b in m.BASES:
        torch.manual_seed(379)
        net=m.Net(state['extra'],b['quantiles']).cuda()
        ema=m.Net(state['extra'],b['quantiles']).cuda()
        ema.load_state_dict(net.state_dict())
        ema.requires_grad_(False)
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        for step in range(1,13):
            net.train()
            out=net(x)
            if b['quantiles']:
                assert (out[:,0]<=out[:,1]).all() and (out[:,1]<=out[:,2]).all()
            loss=m.objective(out,y,b['quantiles'],True)
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();opt.step();m.update_ema(net,ema,step)
        before=m.infer(ema,q)
        path=m.ROOT/('smoke_'+b['name']+'.pt')
        torch.save(ema.state_dict(),path)
        net.load_state_dict(torch.load(path,weights_only=True))
        after=m.infer(net,q)
        np.testing.assert_array_equal(before,after)
        records.append(dict(base=b['name'],steps=12,replay_max_error=0.,loss=float(loss.detach())))
    result=dict(status='PASS',signature=m.signature(),records=records,outer_scores_computed=False,all_fitting_remote=True)
    (m.ROOT/'smoke_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
