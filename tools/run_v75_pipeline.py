"""Develop two distinct branches, freeze, confirm, then build a reviewable release."""
from pathlib import Path
import sys,subprocess,json,os,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v75_experiments as v


def status(stage,**kwargs):b.save(v.OUT/'pipeline_status.json',dict(status='RUNNING',stage=stage,pid=os.getpid(),**kwargs))


def run(script,*args):
    name=Path(script).stem
    with (ROOT/(name+'.log')).open('a') as f:
        subprocess.run([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    import torch
    assert os.name!='nt' and torch.cuda.is_available()
    d=b.build();v.OUT.mkdir(exist_ok=True);v.audit(d)
    tasks=sorted(d['tasks'],key=lambda t:sum((d['tid']==d['tasks'].index(t))&(d['split']==0)))
    status('development')
    with (ROOT/'v75_kernels.log').open('a') as log:
        kernel=subprocess.Popen([sys.executable,'-u',str(ROOT/'admet_v75_experiments.py'),'kernels','--tasks',*tasks],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            v.develop_tree(d,tasks,v.P['selection_parts'])
            status('waiting_kernel_development',kernel_pid=kernel.pid)
            if kernel.wait()!=0:raise RuntimeError('Kernel development failed; see v75_kernels.log')
        except BaseException:
            if kernel.poll() is None:kernel.terminate();kernel.wait()
            raise
    from select_v75 import selection,confirmation
    frozen=selection(d)
    selected=[r for r in frozen['tasks'] if r['selection'] is not None]
    status('confirmation',tasks=[r['task'] for r in selected])
    views=None
    for r in selected:
        task=r['task'];k=d['tasks'].index(task);c=r['selection']
        v.develop_tree(d,[task],v.P['confirmation_parts'])
        if c['kernel'] is not None:
            if views is None:views=v.fingerprints(d)
            for part in v.P['confirmation_parts']:v.fit_kernels(d,views,k,part)
        # Confirmation matches the final ensemble; the control gets equal seeds.
        aux,audit=v.auxiliaries(d,k)
        for part in v.P['confirmation_parts']:
            fit,q=v.rows(d,k,part)
            for family,values in [('structure_tree',None),('conditional_tree',aux)]:
                if family=='conditional_tree' and not c['conditional']:continue
                folder=v.OUT/task/part/family
                trees=json.loads((folder/'inner/metadata.json').read_text())['trees']
                for seed in v.P['final_seeds'][1:]:v.fit_tree(d,k,fit,q,folder/f'outer_{seed}',values,audit if values is not None else None,trees,seed)
    confirmed=confirmation(d)
    if not confirmed['accepted']:
        b.save(v.OUT/'pipeline_status.json',dict(status='NO_CONFIRMED_CANDIDATE',stage='confirmation',signature=v.signature()))
        return
    for stage,script in [('finalize','tools/finalize_v75.py'),('verify','tools/verify_v75_models.py'),('package','tools/package_v75_release.py')]:
        status(stage);run(script)
    b.save(v.OUT/'pipeline_status.json',dict(status='COMPLETE',official_score=None,signature=v.signature()))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        b.save(v.OUT/'pipeline_status.json',dict(status='FAILED',error=str(exc)));raise
