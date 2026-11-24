"""Summarize delayed detection using frozen fine trajectories; no ODE solving."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
G=Path(__file__).resolve().parent.parent; R=G/'runs/ensemble'; O=G/'results/detection_fractions.csv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 marker=json.loads((R/'methods/ensemble_freeze.json').read_text(encoding='utf-8'))
 if marker.get('n_solutions')!=30 or not marker.get('policy_independent'):raise RuntimeError('Incomplete O1 freeze')
 for p,h in marker['files'].items():
  if sha(R/p)!=h:raise RuntimeError('Frozen input changed')
 rows=[]
 for j in range(1,31):
  name=f'U{j:02d}'; chk=json.loads((R/'checkpoints'/f'{name}_strategies.json').read_text())
  for p,h in chk['files'].items():
   if sha(R/p)!=h:raise RuntimeError('Trajectory hash mismatch')
  traj=np.load(R/'results/scenarios'/f'{name}_fine_trajectories.npz');t=traj['time']
  for s in ['NN','HN','LN','NH','NL','HH','HL','LH','LL']:
   x=traj[s];aux=np.trapz(x[:,4:6]/7,t,axis=0);total=x[-1,10:12]-x[0,10:12]
   if np.any(aux<0) or np.any(total<=0) or np.any(aux>total+0.01):raise RuntimeError('Detection flow invalid')
   rows.append([name,s,*aux,*total,*(aux/total)])
 with O.open('x',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['solution','strategy','delayed_GD','delayed_GX','total_reported_GD','total_reported_GX','fraction_GD','fraction_GX']);w.writerows(rows)
 print('wrote',O,'rows',len(rows),'(composite trapezoid on saved 0.025-day event grid)')
if __name__=='__main__':main()
