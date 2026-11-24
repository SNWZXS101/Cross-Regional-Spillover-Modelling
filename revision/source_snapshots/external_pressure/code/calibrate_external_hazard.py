"""Pre-policy M2 10% hazard calibration from frozen baseline, independent solve."""
import os,sys,json,hashlib,datetime
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
HERE=Path(__file__).resolve().parent; M2=HERE.parent; PROJECT=M2.parents[2]
for name in ('temp','numba_cache'):(M2/'logs'/name).mkdir(parents=True,exist_ok=True)
os.environ['NUMBA_CACHE_DIR']=str(M2/'logs/numba_cache');os.environ['TEMP']=os.environ['TMP']=str(M2/'logs/temp')
sys.path.insert(0,str(HERE))
from original_independent_rhs import rhs as rhs12
BASE=PROJECT/'baseline/Model4GDGXcq2x_v6_optimized01/Training-results-26-09-12_003513-set1'
FROZEN=PROJECT/'uniform_30_solution_strategy_v2/results/new_baseline_rollout.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 output=M2/'configs/calibration.json'
 if output.exists():raise RuntimeError('Refusing overwrite of fixed calibration')
 cfg=json.loads((BASE/'config.json').read_text());N=np.array([cfg['N1_raw'],cfg['N2_raw']],float)
 rates=np.stack([np.loadtxt(BASE/(k+'.txt')) for k in ['c1','c2','q1','q2','m1','m2','d1','d2']]); beta=float(np.loadtxt(BASE/'beta.txt'));gamma=float(np.loadtxt(BASE/'gamma.txt'));mu=float(np.loadtxt(BASE/'mu.txt'))
 stored=np.load(FROZEN);old=stored['DOP853']; x0=np.array([N[0],N[1]-5,0,4,0,1,0,0,0,0,0,0,0,0,0,0],float)
 def f(t,z):
  x=z[:12];out=rhs12(t,x,rates,beta,gamma,mu,*N);r=np.array([np.interp(t,np.arange(236),rr) for rr in rates]);j=beta*r[:2]*x[:2]*x[4:6]/N
  return np.r_[out,j,x[:2]]
 values=[x0];y=x0.copy()
 for day in range(236):
  sol=solve_ivp(f,(day,day+1),y,method='DOP853',rtol=1e-11,atol=1e-8,max_step=.125)
  if not sol.success or not np.isfinite(sol.y).all():raise RuntimeError('Baseline solve failure '+str(sol.message))
  y=sol.y[:,-1];values.append(y.copy())
 a=np.stack(values);error=float(np.max(np.abs(a[:,:14]-old)))
 if error>=.05:raise RuntimeError('Frozen original rollout mismatch '+str(error))
 j=a[-1,12:14];areas=a[-1,14:16];nu=.1*j/areas
 if not(np.isfinite(nu).all() and (nu>0).all()):raise RuntimeError('Invalid hazard')
 # Independent composite trapezoid of DOP853 dense daily-node solutions is a
 # coarse diagnostic only. Authoritative integral is the integrated states.
 h=.0125;steps=round(236/h);r=x0.copy();area=np.zeros(2)
 for k in range(steps):
  t=k*h;k1=f(t,r);k2=f(t+h/2,r+h*k1/2);k3=f(t+h/2,r+h*k2/2);k4=f(t+h,r+h*k3);new=r+h*(k1+2*k2+2*k3+k4)/6
  area+=h*(r[:2]+new[:2])/2;r=new
 compare=float(np.max(np.abs(area-areas)/areas));jerr=float(np.max(np.abs(r[12:14]-j)))
 if compare>1e-8 or jerr>=.05:raise RuntimeError('Independent baseline integral check failed')
 files=[FROZEN,BASE/'config.json',BASE/'beta.txt',BASE/'gamma.txt',BASE/'mu.txt',*[BASE/(k+'.txt') for k in ['c1','c2','q1','q2','m1','m2','d1','d2']]]
 result={'passed':True,'fraction':.1,'nu_per_day':nu.tolist(),'baseline_cumulative_all_infections':j.tolist(),'baseline_integral_susceptibles_person_days':areas.tolist(),'calibrated_external_infections_on_original_S':(nu*areas).tolist(),'baseline_saved_14state_max_abs_difference':error,'independent_RK4_area_max_relative_difference':compare,'independent_RK4_J_max_abs_difference':jerr,'integration':'days 0..236, daily rate interpolation, DOP853 rtol 1e-11 atol 1e-8 max_step 0.125, J and susceptible-person-days integrated as auxiliary states; RK4 dt 0.0125 and trapezoidal S-area cross-check','source_sha256':{str(p):sha(p) for p in files},'code_sha256':sha(Path(__file__)),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
 print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
