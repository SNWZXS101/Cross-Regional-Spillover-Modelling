"""Isolated T5 strategy engine; evaluates immutable original 100 members."""
from pathlib import Path
import os,sys,json,hashlib,argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd

T5=Path(__file__).resolve().parents[2]; PROJECT=T5.parent.parent
BASE=PROJECT/'baseline/Model4GDGXcq2x_v6_optimized01/Training-results-26-09-12_003513-set1'
SOURCE=PROJECT/'analysis_runs/sensitivity_100'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(c,m):
 if not c:raise RuntimeError(m)
def setup(group,protocol):
 out=(T5/group).resolve();p=Path(protocol).resolve();require(out.is_relative_to(T5.resolve()) and p.is_relative_to(T5.resolve()),'paths outside T5')
 for d in ['results/scenarios','checkpoints','logs','methods']: (out/d).mkdir(parents=True,exist_ok=True)
 ctx={'project_root':str(PROJECT),'output_dir':str(out),'n_solutions':100,'mode':'fresh','source_analysis':str(SOURCE),'baseline_dir':str(BASE),'risk_protocol':str(p),'workers':1}
 path=out/'methods/run_context.json'
 if path.exists():require(json.loads(path.read_text(encoding='utf-8'))==ctx,'changed context')
 else:path.write_text(json.dumps(ctx,indent=2),encoding='utf-8')
 os.environ['UDE_RUN_CONTEXT']=str(path);os.environ['NUMBA_CACHE_DIR']=str(T5/'shared/cache');os.environ['PYTHONDONTWRITEBYTECODE']='1'
 return out

def evaluate(item):
 from scripts.core.solvers import grid,factors,simulate,dop853,dop853_strategy,classify,PROTOCOL,STRATEGIES,T
 name,outstr=item;out=Path(outstr);marker=out/'checkpoints'/f'{name}_strategies.json';target=out/'results/scenarios'
 if marker.exists():
  data=json.loads(marker.read_text());require(all(sha(out/k)==v for k,v in data['files'].items()),'checkpoint mismatch');return name+' reused'
 source=SOURCE/'solutions'/f'{name}.npz';a=np.load(source);r=a['rates'];b=float(a['beta']);dt=float(a['dt']); original=a['DOP853_daily_states']
 tt=grid(.025);di=np.searchsorted(tt,np.arange(T+1));require(np.array_equal(tt[di],np.arange(T+1)),'missing days')
 rows=[];aud=[];daily={};fine_states={};baseline=None
 original_rows=pd.read_csv(SOURCE/'results/complete_scenarios.csv');org=original_rows[(original_rows.solution==name)&(original_rows.strategy=='NN')].set_index('measure')
 for s in STRATEGIES:
  tag=''.join(s);fac=factors(tt,s);fine,mn,mass=simulate(r,dt,b,tt,fac,.00625);coarse,_,_=simulate(r,dt,b,tt,fac,.0125,False)
  er=float(abs(fine[-1,10:]-coarse[-1,10:]).max());require(mn>=0 and mass<.01 and er<.05,f'{name} {tag} RK4 failed {mn} {mass} {er}')
  dd=fine[di];daily[tag]=dd;fine_states[tag]=fine
  if tag=='NN':
   baseline=dd[-1].copy();require(np.max(abs(dd[:,10:]-original[:,10:]))<.05,f'{name} NN daily mismatch')
   for measure,j in [('reported',10),('all_infections',12)]:require(np.max(abs(dd[-1,j:j+2]-org.loc[measure,['C1','C2']].to_numpy(float)))<.01,'NN original mismatch')
  cross=np.nan;event_cross=np.nan;cross_method='none'
  if tag in ['HH','LL']:
   ind=dop853(r,dt,b,s);cross=float(abs(ind[:,10:]-dd[:,10:]).max());event_cross=cross;cross_method='DOP853 event segments'
   if cross>=.05:
    ind=dop853_strategy(r,dt,b,s);cross=float(abs(ind[:,10:]-dd[:,10:]).max());cross_method='DOP853 each rate interval'
   require(cross<.05,f'{name} {tag} independent solver {cross}')
  selected=fac[:,4:6]<1;duration=np.diff(tt)@selected
  for j in range(2):require(abs(duration[j]-(0 if s[j]=='N' else PROTOCOL['duration']))<1e-7,f'{name} {tag} bad duration')
  require(np.all(fac[:,:4]==1),'c/q modified')
  aud.append(dict(solution=name,strategy=tag,GD_control_days=float(duration[0]),GX_control_days=float(duration[1]),c_q_unchanged=True,beta_unchanged=True,outside_mask_change=0,min_state=mn,mass_error=mass,RK_step_error=er,DOP853_event_cumulative_error=event_cross,DOP853_cumulative_error=cross,DOP853_method=cross_method))
  for measure,j in [('reported',10),('all_infections',12)]:
   c=dd[-1,j:j+2];avoided=baseline[j:j+2]-c
   rows.append(dict(solution=name,strategy=tag,GD_strategy=s[0],GX_strategy=s[1],measure=measure,C1=float(c[0]),C2=float(c[1]),A1=float(avoided[0]),A2=float(avoided[1]),Atotal=float(avoided.sum()),classification=classify(avoided,s),win_win=bool((avoided>.01).all()),beta=b))
 paths=[target/f'{name}_scenarios.csv',target/f'{name}_audit.csv',target/f'{name}_daily_trajectories.npz',target/f'{name}_fine_trajectories.npz']
 pd.DataFrame(rows).to_csv(paths[0],index=False);pd.DataFrame(aud).to_csv(paths[1],index=False)
 np.savez_compressed(paths[2],time=np.arange(T+1),**daily);np.savez_compressed(paths[3],time=tt,**fine_states)
 marker.write_text(json.dumps({'source_sha256':sha(source),'files':{str(x.relative_to(out)):sha(x) for x in paths}},indent=2),encoding='utf-8')
 return name+' complete'

def main():
 p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--protocol',required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 require(a.group in ['P1_duration','P2_moderate','P3_strong','P4_risk'] or (a.smoke and a.group.startswith('smoke_')),'invalid group')
 out=setup(a.group,a.protocol)
 manifest=json.loads((SOURCE/'methods/ensemble_freeze.json').read_text(encoding='utf-8'));require(manifest['n_solutions']==100 and manifest['policy_independent'] is True,'bad original freeze')
 base=Path(manifest['base_path']);require(base.resolve()==SOURCE.resolve(),'freeze base')
 for rel,h in manifest['files'].items():require(sha(base/rel)==h,'source freeze mismatch '+rel)
 source={'freeze_sha256':sha(SOURCE/'methods/ensemble_freeze.json'),'protocol_sha256':sha(a.protocol),'baseline_config_sha256':sha(BASE/'config.json'),'runner_sha256':sha(Path(__file__)),'solver_sha256':sha(T5/'shared/code/scripts/core/solvers.py'),'model_sha256':sha(T5/'shared/code/scripts/core/model.py'),'independent_sha256':sha(T5/'shared/code/scripts/core/independent_rhs.py')}
 prov=out/'methods/source.json'
 if prov.exists():require(json.loads(prov.read_text(encoding='utf-8'))==source,'source or config changed since start')
 else:prov.write_text(json.dumps(source,indent=2),encoding='utf-8')
 names=[f'U{i:02d}' for i in range(1,2 if a.smoke else 101)]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for message in pool.map(evaluate,[(n,str(out)) for n in names]):print(message,flush=True)
 rows=pd.concat([pd.read_csv(out/'results/scenarios'/f'{n}_scenarios.csv') for n in names]);rows.to_csv(out/'results/complete_scenarios.csv',index=False)
 aud=pd.concat([pd.read_csv(out/'results/scenarios'/f'{n}_audit.csv') for n in names]);aud.to_csv(out/'results/strategy_audit.csv',index=False)
 print('done',len(names),flush=True)
if __name__=='__main__':main()
