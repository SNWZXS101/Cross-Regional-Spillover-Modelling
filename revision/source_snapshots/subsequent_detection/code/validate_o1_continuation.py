"""Independent saved-rate RK4/DOP853 acceptance; writes only T2 group."""
from pathlib import Path
import sys,json,importlib.util,os
import numpy as np,pandas as pd
from scipy.integrate import solve_ivp
HERE=Path(__file__).resolve().parents[2];WORK=HERE.parent;ROOT=WORK.parent
def main():
 if len(sys.argv) not in (2,3) or (len(sys.argv)==3 and sys.argv[2]!='--smoke'):raise ValueError('Specify O1 run and optional --smoke')
 group=HERE/'O1_detection';pilot=len(sys.argv)==3;parent=Path(sys.argv[1]).resolve()
 if not parent.is_relative_to((group/'runs'/('smoke' if pilot else 'baseline_continuation')).resolve()):raise RuntimeError('Invalid O1 baseline path')
 meta=json.loads((parent/'run_metadata.json').read_text(encoding='utf-8'))
 if meta.get('state')!='training_finished_independent_validation_required' or meta.get('smoke_only') is not pilot or meta.get('seed')!=27182:raise RuntimeError('O1 training not complete')
 import hashlib
 for key,p in [('data_sha256',ROOT/'Data/input/Data_all_sm.xlsx'),('baseline_config_sha256',ROOT/'baseline/Model4GDGXcq2x_v6_optimized01/Training-results-26-09-12_003513-set1/config.json'),('copied_trainer_sha256',group/'code/ude_training.py'),('wrapper_sha256',group/'code/train_o1_continuation.py')]:
  if hashlib.sha256(p.read_bytes()).hexdigest()!=meta.get(key):raise RuntimeError('O1 training provenance mismatch: '+key)
 runs=list(parent.glob('Training-results-*'))
 if len(runs)!=1:raise RuntimeError('Expected one completed O1 baseline')
 rd=runs[0];cfg=json.loads((rd/'config.json').read_text());rates={k:np.loadtxt(rd/(k+'.txt')) for k in ['c1','c2','q1','q2','m1','m2','d1','d2']};beta=float(np.loadtxt(rd/'beta.txt'));gamma=float(np.loadtxt(rd/'gamma.txt'));mu=float(np.loadtxt(rd/'mu.txt'))
 spec=importlib.util.spec_from_file_location('independent_validator',group/'code/independent_validator.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 df=pd.read_excel(ROOT/'Data/input/Data_all_sm.xlsx');obs=df[['GD','GX']].to_numpy(float);n=len(obs);N1=cfg['N1_raw'];N2=cfg['N2_raw']
 x0=np.array([N1,N2-5,0,4,0,1,0,0,0,0,0,0],float)
 if cfg['I10_raw']!=0 or cfg['I20_raw']!=1:raise RuntimeError('O1 initial condition mismatch')
 if any(len(v)!=n for v in rates.values()):raise RuntimeError('Rate count mismatch')
 for k,v in rates.items():
  if not(np.isfinite(v).all() and np.all(v>=cfg[k+'_bounds'][0]) and np.all(v<=cfg[k+'_bounds'][1])):raise RuntimeError('Bounds: '+k)
 if not cfg['beta_bounds'][0]<=beta<=cfg['beta_bounds'][1]:raise RuntimeError('Beta bound')
 print('Independent RK4 .0125 / DOP853','O1',flush=True)
 # Check every substep, not only the saved daily states.
 h=1/80;rk=[x0];x=x0.copy();min_rk=float(x0.min());mass_rk_internal=0.;warns=[]
 for day in range(n):
  for step in range(80):
   t=day+step*h;k1=mod.rhs(t,x,rates,beta,gamma,mu,N1,N2);k2=mod.rhs(t+h/2,x+h*k1/2,rates,beta,gamma,mu,N1,N2);k3=mod.rhs(t+h/2,x+h*k2/2,rates,beta,gamma,mu,N1,N2);k4=mod.rhs(t+h,x+h*k3,rates,beta,gamma,mu,N1,N2)
   x=x+h*(k1+2*k2+2*k3+k4)/6
   if not np.isfinite(x).all():raise RuntimeError('Nonfinite RK4 state')
   min_rk=min(min_rk,float(x.min()));mass_rk_internal=max(mass_rk_internal,abs(float(x[:10].sum())-(N1+N2)))
  rk.append(x.copy())
 rk=np.stack(rk)
 states=[x0];min_dop=float(x0.min());mass_dop_internal=0.;y=x0.copy()
 for day in range(n):
  sol=solve_ivp(lambda t,x:mod.rhs(t,x,rates,beta,gamma,mu,N1,N2),(day,day+1),y,method='DOP853',rtol=1e-11,atol=1e-8,max_step=.125)
  if not sol.success or not np.isfinite(sol.y).all():raise RuntimeError('DOP853 failure '+str(sol.message))
  min_dop=min(min_dop,float(sol.y.min()));mass_dop_internal=max(mass_dop_internal,float(np.max(np.abs(sol.y[:10].sum(axis=0)-(N1+N2)))));y=sol.y[:,-1];states.append(y.copy())
 dop=np.stack(states);pred1=np.diff(rk[:,10:12],axis=0);pred2=np.diff(dop[:,10:12],axis=0)
 fits={f'{method}_{region}':mod.metrics(obs[:,j],pred[:,j]) for method,pred in [('RK4',pred1),('DOP853',pred2)] for j,region in enumerate(['GD','GX'])}
 err=float(np.max(np.abs(rk[:,10:12]-dop[:,10:12])));mass_rk=float(np.max(np.abs(rk[:,:10].sum(axis=1)-(N1+N2))));mass_dop=float(np.max(np.abs(dop[:,:10].sum(axis=1)-(N1+N2))))
 result={'group':'O1','baseline_dir':str(rd),'beta':beta,'fits':fits,'min_state_RK4_all_substeps':min_rk,'min_state_DOP853_internal':min_dop,'mass_error_RK4_all_substeps':mass_rk_internal,'mass_error_DOP853_internal':mass_dop_internal,'cumulative_RK4_DOP853_max_error':err,'rk4_warnings':warns,'rk4_step_days':.0125,'dop853_rtol':1e-11,'dop853_atol':1e-8,'rates_bounds_checked':True}
 result['accepted']=bool(all(x['R2']>.98 for x in fits.values()) and not warns and min_rk>=0 and min_dop>=0 and mass_rk_internal<.01 and mass_dop_internal<.01 and err<.01)
 out=parent/'validation';out.mkdir(parents=True,exist_ok=False);(out/'baseline_independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out/'baseline_independent_rollouts.npz',rk4=rk,dop853=dop,observed=obs)
 print(json.dumps(result,indent=2),flush=True)
 if not pilot and not result['accepted']:raise SystemExit('Baseline did not pass scientific criteria; retained diagnostics')
if __name__=='__main__':main()
