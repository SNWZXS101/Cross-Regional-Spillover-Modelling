"""Fast deterministic smoke checks of published scientific components; no fitting or interventions."""
from pathlib import Path
import importlib.util
import numpy as np
R=Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
m=load('external_pressure',R/'source_snapshots/external_pressure/code/m2_dynamics.py')
x=np.array([127e6,50e6,0,4,1,1,0,0,0,0,0,0,0,0],float);rates=np.array([3,3,.2,.3,.1,.1,.05,.05]);z=m.rhs14(x,rates,.09,1/7,1/7,[127e6,50e6],[3.35e-8,1.07e-7]);assert np.isfinite(z).all() and abs(z[:10].sum())<1e-8
f=load('unsmoothed_diagnostics',R/'source_snapshots/unsmoothed_data/code/fit_diagnostics.py');y=np.ones(236);y[0]=2;p=y.copy();p[-1]+=2;cfg={'w_daily_raw':1.,'w_daily_sqrt':0.,'w_daily_log':0.,'w_cumulative':0.};d=f.diagnostics(y,p,cfg,1);assert d['weekly_last_days']==5 and abs(d['weekly_WAPE']-2/237)<1e-14
o=load('subsequent_detection',R/'source_snapshots/subsequent_detection/code/independent_validator.py')
rr={k:np.ones(237)*v for k,v in zip(o.RATE_NAMES,rates)};s=x[:12].copy();v=o.rhs(0,s,rr,.09,1/7,1/7,127e6,50e6)
assert abs(v[:10].sum())<1e-8 and np.isclose(v[10],.2*.09*3*s[0]*s[4]/127e6+s[4]/7)
print('PASS: external pressure and detection balance; WAPE includes final five days')
