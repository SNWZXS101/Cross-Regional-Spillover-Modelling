"""Compare event-segment trapezoids against subdividing the stored mesh.

Midpoint state is linear interpolation of stored trajectory, not an additional ODE solution.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
T6=Path(__file__).resolve().parents[1];N=np.array([127000000.,50370000.])
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if not (a.input.resolve().is_relative_to(T6.resolve()) and a.output.resolve().is_relative_to(T6.resolve())): raise RuntimeError('input and output must stay in T6')
 fine=pd.read_csv(a.input/'fine_states_and_instantaneous_flows.csv'); intervals=pd.read_csv(a.input/'fine_interval_flow_integrals.csv');result={}
 for s in ['NN','HH','LL']:
  x=fine[fine.strategy==s];z=intervals[intervals.strategy==s];dt=z.dt.to_numpy();state=x[['S1','S2','I1','I2']].to_numpy();rate=x[['c1','c2','q1','q2','m1','m2','d1','d2']].to_numpy();m=z[['GD_multiplier','GX_multiplier']].to_numpy();
  # beta is fixed across original 100; read exact saved member rather than infer from products.
  root=T6.parent.parent;ident=str(x.solution.iloc[0]);beta=float(np.load(root/'analysis_runs/sensitivity_100/solutions'/f'{ident}.npz')['beta'])
  def fun(st,r):
   infection=beta*r[:,:2]*st[:,:2]*st[:,2:]/N; A=(1-r[:,2:4])*infection;return np.hstack([infection,r[:,2:4]*infection,A,r[:,4:6]*A*m,r[:,6:8]*st[:,2:]*m])
  u=fun(state[:-1],rate[:-1]);v=fun(state[1:],rate[1:]);q=fun((state[:-1]+state[1:])/2,(rate[:-1]+rate[1:])/2)
  coarse=np.sum(.5*(u+v)*dt[:,None],axis=0);half=np.sum(.25*(u+2*q+v)*dt[:,None],axis=0)
  fields=['J_GD','J_GX','C_GD','C_GX','A_GD','A_GX','mA_GD','mA_GX','dI_GD','dI_GX'];r={}
  for i,name in enumerate(fields):r[name]={'coarse':float(coarse[i]),'half_subdivided':float(half[i]),'absolute_difference':float(abs(half[i]-coarse[i])),'relative_difference':float(abs(half[i]-coarse[i])/max(abs(half[i]),1.0))}
  r['exact_C_differences']=[float(coarse[2+j]-(x.iloc[-1][f'C{j+1}']-x.iloc[0][f'C{j+1}'])) for j in range(2)]
  r['exact_J_differences']=[float(coarse[j]-(x.iloc[-1][f'J{j+1}']-x.iloc[0][f'J{j+1}'])) for j in range(2)]
  result[s]=r
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps({'method':'event-separated trapezoid; subdivided with linearly interpolated saved states and rates, not a finer ODE solution or rigorous error bound','source_csv_sha256':hashlib.sha256((a.input/'fine_states_and_instantaneous_flows.csv').read_bytes()).hexdigest(),'results':result},indent=2),encoding='utf-8')
 print(str(a.output))
if __name__=='__main__':main()
