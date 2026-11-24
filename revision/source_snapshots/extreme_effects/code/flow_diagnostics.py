"""Evaluate stored-state flow diagnostics without integrating the model."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
T6=Path(__file__).resolve().parents[1]; ROOT=T6.parent.parent
SRC=ROOT/'analysis_runs/sensitivity_100'; PROTO=ROOT/'uniform_30_solution_strategy_v2/methods/fixed_risk_protocol.json'
BASE=ROOT/'baseline/Model4GDGXcq2x_v6_optimized01/Training-results-26-09-12_003513-set1'
N=np.array([127000000.,50370000.]); names=['c1','c2','q1','q2','m1','m2','d1','d2']; state_names=['S1','S2','Sq1','Sq2','I1','I2','Iq1','Iq2','R1','R2','C1','C2','J1','J2']; regions=['GD','GX']
def require(c,m):
 if not c:raise RuntimeError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def masks(t,strategy,protocol):
 out=np.ones((len(t),2))
 for j in range(2):
  high=np.zeros(len(t),bool)
  for a,b in protocol['regions'][regions[j]]['high_intervals']:high|=(t>=a)&(t<b)
  if strategy[j]=='H':out[:,j]=np.where(high,.2,1.)
  elif strategy[j]=='L':out[:,j]=np.where(~high,.2,1.)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--states',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve();require(out.is_relative_to((T6/'results').resolve()) and not out.exists(),'output must be new T6 results directory'); require(a.states.resolve().is_relative_to(T6.resolve()),'states must be T6 output')
 src=SRC/'solutions'/f'{a.id}.npz'; protocol=json.loads(PROTO.read_text(encoding='utf-8'));cfg=json.loads((BASE/'config.json').read_text()); require(np.array_equal(N,[cfg['N1_raw'],cfg['N2_raw']]),'population mismatch')
 with np.load(src,allow_pickle=False) as data:r=np.array(data['rates']);dt=float(data['dt']);beta=float(data['beta'])
 with np.load(a.states,allow_pickle=False) as data:
  t=np.array(data['time']); x={s:np.array(data[s]) for s in ['NN','HH','LL']}
 require(t[0]==0 and t[-1]==236 and (np.diff(t)>0).all() and np.max(np.diff(t))<=.025000001,'bad time grid')
 ids=np.searchsorted(t,np.arange(237));require(np.array_equal(t[ids],np.arange(237)),'missing daily boundaries');require(all(v.shape==(len(t),14) and np.isfinite(v).all() for v in x.values()),'states shape')
 rates=np.column_stack([np.interp(t,np.arange(len(r))*dt,r[:,j]) for j in range(8)]); gamma=float(np.loadtxt(BASE/'gamma.txt')); durations=np.diff(t);mid=(t[:-1]+t[1:])/2
 out.mkdir(parents=True); complete=[]; daily=[]; interval=[]; summaries=[]
 for s in ['NN','HH','LL']:
  z=x[s];fm=masks(mid,s,protocol);fn=masks(t,s,protocol)
  infection=beta*rates[:,:2]*z[:,:2]*z[:,4:6]/N; A=(1-rates[:,2:4])*infection;report=rates[:,2:4]*infection
  # Interval endpoint trapezoids use a constant intervention factor per event-delimited interval; no across-event averaging.
  contact_node=rates[:,4:6]*A; migration_node=rates[:,6:8]*z[:,4:6]
  contact_avg=.5*(contact_node[:-1]+contact_node[1:])*fm
  migration_avg=.5*(migration_node[:-1]+migration_node[1:])*fm
  af=.5*(A[:-1]+A[1:]);inf=.5*(infection[:-1]+infection[1:]);rep=.5*(report[:-1]+report[1:]);I=.5*(z[:-1,4:6]+z[1:,4:6]);net=(contact_avg+migration_avg)[:,::-1]-(contact_avg+migration_avg)
  for i in range(len(t)):
   row={'solution':a.id,'strategy':s,'day':t[i]}
   for k,name in enumerate(state_names):row[name]=z[i,k]
   for j,name in enumerate(names):row[name]=rates[i,j]
   for j,reg in enumerate(regions):
    row.update({f'{reg}_report_flow':report[i,j],f'{reg}_new_infections_flow':infection[i,j],f'{reg}_A':A[i,j],f'{reg}_mA_right':contact_node[i,j]*fn[i,j],f'{reg}_dI_right':migration_node[i,j]*fn[i,j],f'{reg}_net_inflow_right':(contact_node[i,1-j]+migration_node[i,1-j])*fn[i,1-j]-(contact_node[i,j]+migration_node[i,j])*fn[i,j]})
   complete.append(row)
  for i in range(len(mid)):
   row={'solution':a.id,'strategy':s,'start':t[i],'end':t[i+1],'dt':durations[i],'GD_multiplier':fm[i,0],'GX_multiplier':fm[i,1]}
   for j,reg in enumerate(regions):
    row.update({f'{reg}_new_infections':inf[i,j]*durations[i],f'{reg}_reported_flow':rep[i,j]*durations[i],f'{reg}_unisolated_inflow_A':af[i,j]*durations[i],f'{reg}_outgoing_mA':contact_avg[i,j]*durations[i],f'{reg}_outgoing_dI':migration_avg[i,j]*durations[i],f'{reg}_net_inflow':net[i,j]*durations[i],f'{reg}_recovery_unisolated':gamma*I[i,j]*durations[i]})
   interval.append(row)
  for day in range(236):
   lo,hi=ids[day],ids[day+1];row={'solution':a.id,'strategy':s,'day_start':day,'day_end':day+1}
   for j,reg in enumerate(regions):
    for name,arr in [('new_infections',inf),('reported_flow',rep),('unisolated_inflow_A',af),('outgoing_mA',contact_avg),('outgoing_dI',migration_avg),('net_inflow',net),('recovery_unisolated',gamma*I)]:row[f'{reg}_{name}_integral']=float(np.dot(arr[lo:hi,j],durations[lo:hi]))
    row[f'{reg}_reported_daily_exact']=float(z[hi,10+j]-z[lo,10+j]);row[f'{reg}_infections_daily_exact']=float(z[hi,12+j]-z[lo,12+j])
   daily.append(row)
  for j,reg in enumerate(regions):
   peak=int(np.argmax(z[:,4+j]));summaries.append(dict(solution=a.id,strategy=s,region=reg,reported=float(z[-1,10+j]),all_infections=float(z[-1,12+j]),reported_fraction=float(z[-1,10+j]/N.sum()),all_infections_fraction=float(z[-1,12+j]/N.sum()),peak_unisolated=float(z[peak,4+j]),peak_day=float(t[peak]),final_susceptible=float(z[-1,j]),final_susceptible_fraction=float(z[-1,j]/N[j]),total_contact_export=float(np.dot(contact_avg[:,j],durations)),total_migration_export=float(np.dot(migration_avg[:,j],durations)),total_net_inflow=float(np.dot(net[:,j],durations)),total_A=float(np.dot(af[:,j],durations)),balance_residual=float(z[-1,4+j]-z[0,4+j]-np.dot(af[:,j]+net[:,j]-gamma*I[:,j],durations))))
  summaries.append(dict(solution=a.id,strategy=s,region='total',reported=float(z[-1,10:12].sum()),all_infections=float(z[-1,12:14].sum()),reported_fraction=float(z[-1,10:12].sum()/N.sum()),all_infections_fraction=float(z[-1,12:14].sum()/N.sum())))
 pd.DataFrame(complete).to_csv(out/'fine_states_and_instantaneous_flows.csv',index=False,float_format='%.17g');pd.DataFrame(interval).to_csv(out/'fine_interval_flow_integrals.csv',index=False,float_format='%.17g');pd.DataFrame(daily).to_csv(out/'daily_flow_integrals.csv',index=False,float_format='%.17g');pd.DataFrame(summaries).to_csv(out/'strategy_burdens_and_peaks.csv',index=False,float_format='%.17g')
 peaks=[];start=pd.Timestamp('2022-01-29')
 for s in ['NN','HH','LL']:
  z=x[s]
  for j,reg in [(0,'GD'),(1,'GX'),(2,'total')]:
   I=z[:,4+j] if j<2 else z[:,4]+z[:,5];C=z[ids,10+j] if j<2 else z[ids,10]+z[ids,11];J=z[ids,12+j] if j<2 else z[ids,12]+z[ids,13]
   for metric,arr in [('unisolated_infectious_stock',I),('daily_reported_cases',np.diff(C)),('daily_all_infections',np.diff(J))]:
    k=int(np.argmax(arr));day=float(t[k]) if metric=='unisolated_infectious_stock' else float(k)
    peaks.append(dict(solution=a.id,strategy=s,region=reg,metric=metric,value=float(arr[k]),day=day,date=(start+pd.Timedelta(days=int(day))).date().isoformat(),resolution='event-aligned ≤0.025 day' if metric=='unisolated_infectious_stock' else 'one-day interval beginning at day'))
 pd.DataFrame(peaks).to_csv(out/'peaks_by_region_and_metric.csv',index=False,float_format='%.17g')
 hh,ll=x['HH'],x['LL'];delta=ll-hh;pd.DataFrame({'day':t,'reported_gap_GD':delta[:,10],'reported_gap_GX':delta[:,11],'reported_gap_total':delta[:,10]+delta[:,11],'infections_gap_GD':delta[:,12],'infections_gap_GX':delta[:,13],'infections_gap_total':delta[:,12]+delta[:,13],'I_gap_GD':delta[:,4],'I_gap_GX':delta[:,5],'S_gap_GD':delta[:,0],'S_gap_GX':delta[:,1]}).to_csv(out/'HH_LL_gap_time_series.csv',index=False,float_format='%.17g')
 fig,ax=plt.subplots(3,2,figsize=(12,10));col={'NN':'#506575','HH':'#007d82','LL':'#bd853c'}
 for s in ['NN','HH','LL']:
  z=x[s];ax[0,0].plot(t,z[:,10]+z[:,11],label=s,color=col[s]);ax[0,1].plot(t,z[:,4]+z[:,5],color=col[s]);ax[1,0].plot(t,z[:,4],color=col[s]);ax[1,1].plot(t,z[:,5],color=col[s]);ax[2,0].plot(t,z[:,0]/N[0],color=col[s]);ax[2,1].plot(t,z[:,1]/N[1],color=col[s])
 for q,label in zip(ax.flat,['cumulative reported total','unisolated I total','GD unisolated I','GX unisolated I','GD susceptible / N','GX susceptible / N']):q.set_xlabel('day');q.set_ylabel(label)
 ax[0,0].legend(frameon=False);fig.tight_layout();fig.savefig(out/'trajectory_diagnostic.png',dpi=180);plt.close(fig)
 paths=[src,PROTO,BASE/'config.json',a.states]+list(out.glob('*.csv'))+[out/'trajectory_diagnostic.png'];(out/'sha256_manifest.json').write_text(json.dumps({str(p):sha(p) for p in paths},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'solution':a.id,'time_points':len(t),'output':str(out),'max_abs_balance_residual':max(abs(float(s['balance_residual'])) for s in summaries if s['region']!='total')},ensure_ascii=False))
if __name__=='__main__':main()
