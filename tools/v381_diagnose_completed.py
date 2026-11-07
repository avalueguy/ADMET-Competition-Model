"""Posthoc descriptive diagnostics for completed fits; never change selection."""
from pathlib import Path
import json
import joblib
import numpy as np
from scipy.stats import ks_2samp,spearmanr
import v381_oof_pareto as m


def main():
    assert json.loads((m.ROOT/'completion_receipt.json').read_text())['status']=='PASS'
    d=m.load();d.update(dict(np.load(m.ROOT/'stereo_keys.npz')))
    dev=json.loads((m.OUT/'development.json').read_text())
    conf=json.loads((m.OUT/'confirmation.json').read_text())
    records=[]
    for record in dev['records']:
        t=record['task'];c=record['best']['config'];parts=[]
        allowed=('fold0','fold1','fold2','valid') if any(r['task']==t for r in conf['records']) else ('fold0','fold1')
        for part in allowed:
            a=m.ctrl(t,part);y=d['y'][a['query']];yf=d['y'][a['fit']]
            pred=np.load(m.OUT/'tasks'/t/part/c['name']/'predictions.npz')['p']
            w=record['best']['weight'];blend=(1-w)*a['p']+w*pred
            abs_old=np.abs(y-a['p']);abs_new=np.abs(y-blend)
            cutoff=max(1,int(np.ceil(.1*len(y))));ix=np.argsort(-abs_old)[:cutoff]
            state=joblib.load(m.OUT/'tasks'/t/part/c['name']/'state.joblib')
            oof=joblib.load(m.OUT/'oof'/t/part/'state.joblib')
            oof_pred=oof['oof']@state['weights']+state['bias']
            # Oracle is a measurement diagnostic, never a model to submit.
            keys=d['isomeric'][d['mi'][a['fit']]]
            unique=np.unique(keys);irreducible=0.;duplicate_groups=0;duplicate_rows=0
            for key in unique:
                values=yf[keys==key]
                if len(values)>1:
                    duplicate_groups+=1;duplicate_rows+=len(values)
                    irreducible+=np.abs(values-np.median(values)).sum()
            item=dict(part=part,fit_rows=len(yf),query_rows=len(y),fit_groups=len(np.unique(d['rg'][a['fit']])),
                      query_groups=len(np.unique(d['rg'][a['query']])),fit_quantiles=np.quantile(yf,[0,.1,.5,.9,1]).tolist(),
                      query_quantiles=np.quantile(y,[0,.1,.5,.9,1]).tolist(),target_ks_descriptive=float(ks_2samp(yf,y).statistic),
                      baseline_MAE=float(abs_old.mean()),candidate_MAE=float(abs_new.mean()),
                      standalone_expert_MAE=float(np.abs(y-pred).mean()),innerOOF_stack_MAE=float(np.abs(yf-oof_pred).mean()),
                      baseline_error_fraction_top10percent_rows=float(abs_old[ix].sum()/abs_old.sum()),
                      candidate_error_on_baseline_hard10percent=float(abs_new[ix].mean()),
                      stack_weights=state['weights'].tolist(),stack_bias=float(state['bias']),
                      isomeric_duplicate_groups=duplicate_groups,isomeric_duplicate_rows=duplicate_rows,
                      fit_groupwise_oracle_duplicate_absolute_loss_per_fit_row=float(irreducible/len(yf)))
            if t in m.TASKS[5:]:item['rank_gain']=float(spearmanr(y,blend).statistic-spearmanr(y,a['score_p']).statistic)
            parts.append(item)
        records.append(dict(task=t,parts=parts))
    value=dict(signature=m.signature(),status='DESCRIPTIVE_ONLY',records=records,no_fitting=True,no_reselection=True,
               warning='Inner OOF stack MAE is evaluated on labels used to fit stack weights, so is meta-training error, not unbiased stacked validation. KS is descriptive after reused query labels. Duplicate oracle requires exact same output within isomeric keys; it is not global Bayes error or evidence that noisy labels should be removed.')
    m.save(m.ROOT/'diagnostics.json',value)
    print(json.dumps(dict(status=value['status'],tasks=len(records),no_fitting=True,no_reselection=True)))


if __name__=='__main__':main()
