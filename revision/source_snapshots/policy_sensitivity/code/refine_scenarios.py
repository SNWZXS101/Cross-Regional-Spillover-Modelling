"""Independent higher-precision diagnostic for explicitly named saved scenarios."""
from pathlib import Path
import argparse,hashlib,json,os
import numpy as np
T5=Path(__file__).resolve().parents[2];PROJECT=T5.parent.parent;SOURCE=PROJECT/'analysis_runs/sensitivity_100'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--solution',required=True);p.add_argument('--strategy',choices=['NN','HN','LN','NH','NL','HH','HL','LH','LL'],required=True);a=p.parse_args()
 out=(T5/a.group).resolve()
 if not out.is_relative_to(T5.resolve()) or not (out/'checkpoints'/f'{a.solution}_strategies.json').is_file():raise RuntimeError('invalid or incomplete source')
 checkpoint=json.loads((out/'checkpoints'/f'{a.solution}_strategies.json').read_text(encoding='utf-8'))
 for rel,digest in checkpoint['files'].items():
  target=(out/rel).resolve()
  if not target.is_relative_to(out) or not target.is_file() or sha(target)!=digest:raise RuntimeError('checkpoint hash mismatch')
 context=json.loads((out/'methods/run_context.json').read_text(encoding='utf-8')); provenance=json.loads((out/'methods/source.json').read_text(encoding='utf-8'))
 if sha(Path(context['risk_protocol']))!=provenance['protocol_sha256']:raise RuntimeError('protocol changed')
 os.environ['UDE_RUN_CONTEXT']=str(out/'methods/run_context.json');os.environ['NUMBA_CACHE_DIR']=str(T5/'shared/cache')
 from scripts.core.solvers import grid,factors,simulate,dop853_strategy,T
 source=SOURCE/'solutions'/f'{a.solution}.npz'
 if sha(source)!=checkpoint['source_sha256']:raise RuntimeError('member source changed')
 v=np.load(source);rate=v['rates'];dt=float(v['dt']);beta=float(v['beta']);tt=grid(.025);fac=factors(tt,tuple(a.strategy));ix=np.searchsorted(tt,np.arange(T+1))
 x,mn,mass=simulate(rate,dt,beta,tt,fac,.0015625);coarse,_,_=simulate(rate,dt,beta,tt,fac,.003125,False);ind=dop853_strategy(rate,dt,beta,tuple(a.strategy))
 if mn<0 or mass>=.01 or not np.isfinite(x).all():raise RuntimeError('state or mass failure')
 error=float(abs(x[-1,10:]-coarse[-1,10:]).max());inderr=float(abs(x[ix,10:]-ind[:,10:]).max())
 dest=out/'results/refinement';dest.mkdir(parents=True,exist_ok=True);npz=dest/f'{a.solution}_{a.strategy}_refined.npz';js=dest/f'{a.solution}_{a.strategy}_refined.json'
 if npz.exists() or js.exists():raise RuntimeError('refusing overwrite')
 np.savez_compressed(npz,time=tt,fine=x,independent_daily=ind)
 result={'solution':a.solution,'strategy':a.strategy,'fine_step_days':.0015625,'coarse_step_days':.003125,'fine_coarse_cumulative_error':error,'fine_DOP853_cumulative_error':inderr,'min_state':float(mn),'mass_error':float(mass),'source_sha256':sha(source),'original_checkpoint_sha256':sha(out/'checkpoints'/f'{a.solution}_strategies.json'),'refined_npz_sha256':sha(npz),'acceptance':bool(error<.05 and inderr<.05)}
 js.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
 if not result['acceptance']:raise RuntimeError('refined solver consistency failed; evidence retained')
if __name__=='__main__':main()
