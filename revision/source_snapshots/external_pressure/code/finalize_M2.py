"""Commit M2 only after policy-blind freeze, nine paired scenarios, and audits."""
from pathlib import Path
import json,hashlib,os,datetime
import numpy as np,pandas as pd
M=Path(__file__).resolve().parent.parent;T3=M.parent;R=M/'runs/ensemble'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=M/'reports/T3_group_complete.json'
 if out.exists():raise RuntimeError('Refuse replacement of group marker')
 freeze=R/'methods/ensemble_freeze.json';d=json.loads(freeze.read_text());ids=[f'U{i:02}' for i in range(1,31)]
 if d.get('n_solutions')!=30 or d.get('member_ids')!=ids or d.get('policy_independent') is not True or Path(d['base_path']).resolve()!=R.resolve():raise RuntimeError('Invalid freeze')
 for f,h in d['files'].items():
  if sha(R/f)!=h:raise RuntimeError('Frozen file changed '+f)
 scenarios=R/'results/complete_scenarios.csv';audit=R/'results/strategy_audit.csv';m=pd.read_csv(scenarios);a=pd.read_csv(audit);strategies=['NN','HN','LN','NH','NL','HH','HL','LH','LL']
 if len(m)!=540 or set(zip(m.solution,m.strategy,m.measure))!={(i,s,k) for i in ids for s in strategies for k in ['reported','all_infections']}:raise RuntimeError('Incomplete scenarios')
 if len(a)!=270 or set(zip(a.solution,a.strategy))!={(i,s) for i in ids for s in strategies} or not np.isfinite(a.RK_step_error).all() or (a.RK_step_error>=.05).any():raise RuntimeError('Invalid strategy audit')
 for ident in ids:
  marker=R/'checkpoints'/(ident+'_strategies.json');transaction=json.loads(marker.read_text());files=transaction['files']
  for rel,h in files.items():
   if sha(R/rel)!=h:raise RuntimeError('Scenario checkpoint hash changed')
  normalized={k.replace('\\','/') for k in files}
  for suffix in ['_scenarios.csv','_daily_trajectories.npz','_fine_trajectories.npz','_external_cumulative.npz','_external.csv','_audit.csv']:
   if f'results/scenarios/{ident}{suffix}' not in normalized:raise RuntimeError('Missing scenario artifact '+ident+suffix)
  daily=np.load(R/'results/scenarios'/(ident+'_daily_trajectories.npz'));fine=np.load(R/'results/scenarios'/(ident+'_fine_trajectories.npz'));times=fine['time'];indices=np.searchsorted(times,np.arange(237))
  for s in strategies:
   if daily[s].shape!=(237,14) or fine[s].shape!=(len(times),14) or not np.array_equal(daily[s],fine[s][indices]):raise RuntimeError('Trajectory mismatch '+ident+s)
 ext=R/'results/external_infections.csv';e=pd.read_csv(ext)
 if len(e)!=270 or set(zip(e.solution,e.strategy))!={(i,s) for i in ids for s in strategies} or not np.isfinite(e[['external_fraction_GD','external_fraction_GX']].to_numpy()).all():raise RuntimeError('External incidence report incomplete')
 quad=M/'results/external_quadrature_verification.json';base=M/'results/baseline_independent_validation.json';integrity=T3/'audit_protection/protected_check_M2_complete.json'
 for p in [quad,base,integrity]:
  x=json.loads(p.read_text(encoding='utf-8'))
  if not(x.get('passed') is True or x.get('accepted') is True):raise RuntimeError('Required validation did not pass '+str(p))
 paths={'freeze':freeze,'complete_scenarios':scenarios,'strategy_audit':audit,'ensemble_manifest':R/'results/ensemble_manifest.csv','ensemble_distances':R/'results/ensemble_distances.csv','baseline_validation':base,'source_integrity':integrity,'external_quadrature_validation':quad,'external_infections':ext,'policy_summary':M/'results/policy/policy_effect_summary.csv','policy_rankings':M/'results/policy/pairwise_rankings.csv'}
 for name,p in paths.items():
  if not p.is_file():raise RuntimeError('Missing artifact '+name)
 with out.open('x',encoding='utf-8') as f:json.dump({'completed':True,'n_solutions':30,'policy_selection_blind':True,'artifacts':{k:{'path':str(v.resolve()),'sha256':sha(v)} for k,v in paths.items()},'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f,indent=2)
 print('M2 group committed',out)
if __name__=='__main__':main()
