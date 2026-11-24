"""Validate completed M2 UDE by independent 16-state RK4 and DOP853."""
from pathlib import Path
import os,sys,json,hashlib,datetime
import numpy as np,pandas as pd
from scipy.integrate import solve_ivp
HERE=Path(__file__).resolve().parent;M2=HERE.parent;PROJECT=M2.parents[2]
from m2_dynamics import rhs14
NAMES=['c1','c2','q1','q2','m1','m2','d1','d2']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if len(sys.argv)!=2:raise ValueError('Pass completed manual M2 run directory')
 run=Path(sys.argv[1]).resolve();parent=(M2/'runs/pycharm_manual').resolve()
 if run.parent!=parent:raise RuntimeError('Run directory outside M2 manual area')
 meta=json.loads((run/'revision_run_metadata.json').read_text(encoding='utf-8'));cal=M2/'configs/calibration.json';info=json.loads(cal.read_text(encoding='utf-8'))
 if meta.get('run_state')!='completed_training_requires_independent_acceptance' or meta.get('smoke_only') is not False or meta.get('seed')!=31415:raise RuntimeError('Incomplete M2 training')
 for path,key in [(cal,'calibration_sha256'),(HERE/'train_M2.py','script_sha256'),(PROJECT/'Data/input/Data_all_sm.xlsx','data_sha256')]:
  if sha(path)!=meta[key]:raise RuntimeError('Training source hash mismatch: '+str(path))
 roots=list(run.glob('Training-results-*'))
 if len(roots)!=1:raise RuntimeError('Expected exactly one training output')
 base=roots[0];cfg=json.loads((base/'config.json').read_text(encoding='utf-8'));nu=np.array(info['nu_per_day']);N=np.array([cfg['N1_raw'],cfg['N2_raw']]);b=float(np.loadtxt(base/'beta.txt'));gamma=float(np.loadtxt(base/'gamma.txt'));mu=float(np.loadtxt(base/'mu.txt'))
 if not np.array_equal(nu,np.array([cfg['nu1_external'],cfg['nu2_external']])) or not(cfg['beta_bounds'][0]<=b<=cfg['beta_bounds'][1]) or gamma!=cfg['gamma_fixed'] or mu!=cfg['mu_fixed']:raise RuntimeError('M2 baseline fixed parameters mismatch')
 rates=np.column_stack([np.loadtxt(base/(k+'.txt')) for k in NAMES]);lo=np.array([cfg[k+'_bounds'][0] for k in NAMES]);hi=np.array([cfg[k+'_bounds'][1] for k in NAMES])
 if rates.shape!=(236,8) or not np.isfinite(rates).all() or np.any(rates<lo) or np.any(rates>hi):raise RuntimeError('M2 bounds fail')
 x0=np.r_[N[0],N[1]-5,0.,4.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
 def fun(t,x):
  r=np.array([np.interp(t,np.arange(236),rates[:,i]) for i in range(8)]);return np.r_[rhs14(x,r,b,gamma,mu,N,nu),nu*x[:2]]
 h=1/80;x=x0.copy();values=[x.copy()];minimum=0.;masserr=0.
 for k in range(236*80):
  t=k*h;k1=fun(t,x);k2=fun(t+h/2,x+h*k1/2);k3=fun(t+h/2,x+h*k2/2);k4=fun(t+h,x+h*k3);x=x+h*(k1+2*k2+2*k3+k4)/6
  if not np.isfinite(x).all():raise RuntimeError('M2 RK4 not finite')
  minimum=min(minimum,float(x[:10].min()));masserr=max(masserr,float(abs(x[:10].sum()-N.sum())))
  if (k+1)%80==0:values.append(x.copy())
 rk=np.stack(values);y=x0.copy();daily=[y.copy()];mind=0.;massd=0.
 for day in range(236):
  sol=solve_ivp(fun,(day,day+1),y,method='DOP853',rtol=1e-11,atol=1e-8,max_step=.125)
  if not sol.success or not np.isfinite(sol.y).all():raise RuntimeError('M2 DOP853 failure')
  mind=min(mind,float(sol.y[:10].min()));massd=max(massd,float(np.max(abs(sol.y[:10].sum(axis=0)-N.sum()))));y=sol.y[:,-1];daily.append(y.copy())
 dop=np.stack(daily);obs=pd.read_excel(PROJECT/'Data/input/Data_all_sm.xlsx')[['GD','GX']].to_numpy(float)
 def metrics(pred):
  return [{'R2':float(1-np.sum((pred[:,j]-obs[:,j])**2)/np.sum((obs[:,j]-obs[:,j].mean())**2)),'RMSE':float(np.sqrt(np.mean((pred[:,j]-obs[:,j])**2)))} for j in range(2)]
 rkfit=metrics(np.diff(rk[:,10:12],axis=0));dpfit=metrics(np.diff(dop[:,10:12],axis=0));err=float(np.max(abs(rk[:,10:16]-dop[:,10:16])));fraction=dop[-1,14:16]/dop[-1,12:14]
 ok=min(z['R2'] for z in rkfit+dpfit)>.98 and minimum>=0 and mind>=0 and masserr<.01 and massd<.01 and err<.01 and np.isfinite(fraction).all() and np.all((fraction>=0)&(fraction<=1))
 result={'accepted':bool(ok),'group':'M2','baseline_dir':str(base),'training_run':str(run),'calibration_sha256':sha(cal),'beta':b,'nu_per_day':nu.tolist(),'RK4':rkfit,'DOP853':dpfit,'min_state_RK4':minimum,'min_state_DOP853':mind,'mass_error_RK4':masserr,'mass_error_DOP853':massd,'case_and_diagnostic_max_error':err,'external_cumulative':dop[-1,14:16].tolist(),'all_infections_cumulative':dop[-1,12:14].tolist(),'external_fraction':fraction.tolist(),'time_horizon_days':236,'nu_is_fixed_and_flow_responds_to_S':True,'source_sha256':{str(p):sha(p) for p in [run/'revision_run_metadata.json',base/'config.json',base/'beta.txt',*[base/(k+'.txt') for k in NAMES]]},'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 dest=M2/'results/baseline_independent_validation.json';arr=M2/'results/baseline_independent_rollouts.npz'
 if dest.exists() or arr.exists():raise RuntimeError('Refusing to overwrite prior validation')
 with arr.open('xb') as f:np.savez_compressed(f, RK4=rk,DOP853=dop,observed=obs,time=np.arange(237))
 with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
 print(json.dumps(result,indent=2))
 if not ok:raise SystemExit('M2 baseline rejected; diagnostics preserved')
if __name__=='__main__':main()
