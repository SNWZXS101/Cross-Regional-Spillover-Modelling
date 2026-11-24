"""Isolated B3 maximin, immutable freeze, and nine-strategy entry point.

Use --pilot only for a disposable one-member safety run. Formal execution uses
the fixed 30-member configuration and is blocked until B1/B2 have completed.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,datetime,subprocess,platform,importlib.metadata,csv
import numpy as np
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent;EXP=HERE.parent;WORK=EXP.parents[1];PROJECT=WORK.parent
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def verify_marker(root,marker):
 data=json.loads(marker.read_text(encoding='utf-8'))
 for rel,h in data['files'].items():
  if sha(root/rel)!=h:raise RuntimeError('Checkpoint SHA mismatch '+rel)
 return data
def verify_group_barrier(t2,dirname):
 """Do not trust a completion flag without validating committed artifacts."""
 group=t2/dirname;marker=group/'reports/T2_group_complete.json'
 if not marker.is_file():raise RuntimeError('B1/B2 completion barrier unmet: '+str(marker))
 meta=json.loads(marker.read_text(encoding='utf-8'))
 if meta.get('completed') is not True or meta.get('n_solutions')!=30 or meta.get('policy_selection_blind') is not True:raise RuntimeError('Group completion metadata invalid: '+dirname)
 run=group/'runs/ensemble';checks={
  'freeze_sha256':run/'methods/ensemble_freeze.json',
  'scenario_sha256':run/'results/complete_scenarios.csv',
  'strategy_audit_sha256':run/'results/strategy_audit.csv',
  'policy_summary_sha256':group/'results/policy/policy_effect_summary.csv',
  'policy_ranking_sha256':group/'results/policy/pairwise_rankings.csv',
  'baseline_validation_sha256':group/'results/baseline_independent_validation.json',
  'original_integrity_sha256':t2/'logs'/('original_integrity_'+dirname+'_complete.json')}
 for key,path in checks.items():
  if not path.is_file() or meta.get(key)!=sha(path):raise RuntimeError('Group artifact hash mismatch: '+dirname+'/'+key)
 freeze=json.loads(checks['freeze_sha256'].read_text(encoding='utf-8'));ids=[f'U{i:02d}' for i in range(1,31)]
 if freeze.get('n_solutions')!=30 or freeze.get('member_ids')!=ids or Path(freeze['base_path']).resolve()!=run.resolve():raise RuntimeError('Group frozen member set invalid: '+dirname)
 for rel,h in freeze['files'].items():
  p=run/rel
  if not p.is_file() or sha(p)!=h:raise RuntimeError('Group frozen artifact changed: '+dirname+'/'+rel)
 for name in ids:
  if not any(k.replace('\\','/')==f'solutions/{name}.npz' for k in freeze['files']):raise RuntimeError('Group frozen member missing: '+name)
 for key,expected in [('scenario_sha256',540),('strategy_audit_sha256',270),('policy_summary_sha256',8),('policy_ranking_sha256',72)]:
  with checks[key].open(newline='',encoding='utf-8-sig') as stream:
   rows=list(csv.DictReader(stream))
  if len(rows)!=expected:raise RuntimeError('Group row count differs: '+dirname+'/'+key)
  strategies=['NN','HN','LN','NH','NL','HH','HL','LH','LL'];measures=['reported','all_infections']
  if key=='scenario_sha256' and {(r['solution'],r['strategy'],r['measure']) for r in rows}!={(i,s,m) for i in ids for s in strategies for m in measures}:raise RuntimeError('Group scenario keys invalid: '+dirname)
  if key=='strategy_audit_sha256' and {(r['solution'],r['strategy']) for r in rows}!={(i,s) for i in ids for s in strategies}:raise RuntimeError('Group strategy audit keys invalid: '+dirname)
  if key=='policy_summary_sha256' and {(r['measure'],r['functional']) for r in rows}!={(m,f) for m in measures for f in ['A_GD_HN','A_GX_NH','G_HH_minus_LL','A_total_HH']}:raise RuntimeError('Group summary keys invalid: '+dirname)
  if key=='policy_ranking_sha256' and {(r['measure'],r['strategy_A'],r['strategy_B']) for r in rows}!={(m,a,b) for m in measures for i,a in enumerate(strategies) for b in strategies[i+1:]}:raise RuntimeError('Group ranking keys invalid: '+dirname)
 if json.loads(checks['original_integrity_sha256'].read_text(encoding='utf-8')).get('passed') is not True or json.loads(checks['baseline_validation_sha256'].read_text(encoding='utf-8')).get('accepted') is not True:raise RuntimeError('Group baseline or source integrity failed: '+dirname)
 return meta
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--strategy',action='store_true');args=ap.parse_args()
 if not args.pilot:
  for dirname in ['B1_alt_baseline_A','B2_alt_baseline_B']:
   verify_group_barrier(EXP.parent,dirname)
 root=EXP/'runs'/('smoke_v7' if args.pilot else 'main');root.mkdir(parents=True,exist_ok=True)
 if not root.resolve().is_relative_to(WORK.resolve()):raise RuntimeError('Refuse output outside Revise_work')
 for name in ['solutions','methods','results','checkpoints','logs','figures','numba_cache','temp']:(root/name).mkdir(exist_ok=True)
 cfg=dict(mode='fresh',n_solutions=1 if args.pilot else 30,seed=31415,n_starts=1 if args.pilot else 3,adaptive_start_rounds=[1] if args.pilot else [3,3,6],max_iterations=1 if args.pilot else 60,workers=1,output_dir=str(root),baseline_dir='Model4GDGXcq2x_v6_optimized01/Training-results-26-09-12_003513-set1',risk_protocol='uniform_30_solution_strategy_v2/methods/fixed_risk_protocol.json',source_analysis='uniform_30_solution_strategy_v2',project_root=str(PROJECT))
 config=root/'context.json'
 if config.exists() and json.loads(config.read_text(encoding='utf-8'))!=cfg:raise RuntimeError('Existing context differs')
 if not config.exists():write(config,cfg)
 os.environ.update(UDE_RUN_CONTEXT=str(config),PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(root/'numba_cache'),TEMP=str(root/'temp'),TMP=str(root/'temp'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 sys.dont_write_bytecode=True;sys.path.insert(0,str(HERE))
 import scripts.generate_uniform_ensemble as original
 from pilot_b3 import B3Search
 # The search helper inherits this run's isolated context and cache paths.
 os.environ['UDE_RUN_CONTEXT']=str(config)
 # Globals in copied solvers are initialized during import: context must match.
 if original.ROOT.resolve()!=root.resolve():raise RuntimeError('Copied solver context points outside run')
 inputs={}
 for p in [PROJECT/'Data/input/Data_all_sm.xlsx',PROJECT/'uniform_30_solution_strategy_v2/methods/fixed_risk_protocol.json',*[PROJECT/cfg['baseline_dir']/f for f in ['config.json','beta.txt','gamma.txt','mu.txt','c1.txt','c2.txt','q1.txt','q2.txt','m1.txt','m2.txt','d1.txt','d2.txt']]]:inputs[str(p)]=sha(p)
 code={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*.py')) if '__pycache__' not in p.parts}
 packages={}
 for package in ['numpy','scipy','pandas','numba','torch']:
  try:packages[package]=importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:packages[package]=None
 supporting={str(p.relative_to(WORK)):sha(p) for p in [WORK/'shared/code/summarize_policy.py',WORK/'T2_ensemble_robustness/verify_original_sources.py']}
 metadata={'inputs':inputs,'code':code,'supporting_code':supporting,'environment':{'python':platform.python_version(),'platform':platform.platform(),'packages':packages},'config':cfg,'beta_bounds':[.005,.2],'distance':'original eight-function standardized distance, beta excluded','policy_blind':True,'pilot':args.pilot}
 source=root/'methods/source_hashes.json'
 if source.exists():
  if json.loads(source.read_text(encoding='utf-8'))!=metadata:raise RuntimeError('Source/config changed; refuse unsafe resume')
 else:write(source,metadata)
 def protect(phase):
  verifier=WORK/'T2_ensemble_robustness/verify_original_sources.py'
  run=subprocess.run([sys.executable,'-B',str(verifier),'--label',f'B3_{"pilot" if args.pilot else "main"}_{phase}'],cwd=WORK,env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8')
  (root/'logs'/('source_integrity_'+phase+'.log')).write_text(run.stdout+run.stderr,encoding='utf-8')
  if run.returncode:raise RuntimeError('Original source integrity failure: '+phase)
 if args.strategy:
  freeze=root/'methods/ensemble_freeze.json'
  if not freeze.exists():raise RuntimeError('Freeze required before any strategy run')
  for rel,h in json.loads(freeze.read_text(encoding='utf-8'))['files'].items():
   if sha(root/rel)!=h:raise RuntimeError('Frozen output changed '+rel)
  group_marker=EXP/'reports/T2_group_complete.json'
  if not args.pilot and group_marker.exists():
   complete=json.loads(group_marker.read_text(encoding='utf-8'))
   for name,item in complete['artifacts'].items():
    p=Path(item['path'])
    if not p.is_file() or sha(p)!=item['sha256']:raise RuntimeError('Committed group artifact changed '+name)
   print('B3 completed group already committed and verified; no outputs rewritten');return
  import scripts.run_strategy_matrix as strategy
  strategy.strategy_fresh()
  summary=WORK/'shared/code/summarize_policy.py'
  subprocess.run([sys.executable,'-B',str(summary),'--scenarios',str(root/'results/complete_scenarios.csv'),'--strategy-audit',str(root/'results/strategy_audit.csv'),'--output',str(root/'results/policy_summary')],cwd=WORK,env=os.environ.copy(),check=True)
  protect('strategies_complete')
  if not args.pilot:
   artifact_paths={'freeze':freeze,'scenarios':root/'results/complete_scenarios.csv','strategy_audit':root/'results/strategy_audit.csv','policy_summary':root/'results/policy_summary/policy_effect_summary.csv','policy_rankings':root/'results/policy_summary/pairwise_rankings.csv','policy_effects':root/'results/policy_summary/policy_effects.csv','source_integrity':WORK/'T2_ensemble_robustness/logs/original_integrity_B3_main_strategies_complete.json'}
   artifacts={name:dict(path=str(path),sha256=sha(path)) for name,path in artifact_paths.items()}
   write(group_marker,dict(completed=True,n_solutions=cfg['n_solutions'],beta_variable=True,policy_selection_blind=True,artifacts=artifacts))
  return
 freeze=root/'methods/ensemble_freeze.json'
 if freeze.exists():raise RuntimeError('Frozen ensemble cannot be modified')
 ref=original.UniformSearch([np.zeros((original.T*4+1,8))]);baseline=ref.reference;bank=[];manifest=[]
 for k in range(1,cfg['n_solutions']+1):
  member=f'U{k:02d}';marker=root/'checkpoints'/(member+'_accepted.json')
  if marker.exists():
   verify_marker(root,marker);z=np.load(root/'solutions'/(member+'.npz'));bank.append(z['rates']);manifest.append(json.loads((root/'solutions'/(member+'.json')).read_text(encoding='utf-8')));continue
  bank_ref=bank if bank else [baseline];context=dict(member=member,source_digest=digest(metadata),bank_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in bank_ref]);accepted=None;rounds=[];trials_done=0;evals=0
  for round_no,size in enumerate(cfg['adaptive_start_rounds'],1):
   proposals=[]
   for trial in range(trials_done,trials_done+size):
    checkpoint=root/'checkpoints'/f'{member}_start{trial}.json'
    if checkpoint.exists():
     saved=json.loads(checkpoint.read_text(encoding='utf-8'));payload=saved.pop('payload_sha256')
     if digest(saved)!=payload or saved['context']!=context:raise RuntimeError('Start checkpoint mismatch')
    else:
     obj=B3Search([r[::10] for r in bank_ref]);phase=np.random.default_rng(np.random.SeedSequence([cfg['seed'],k,trial,3103])).uniform(-.002,.002)
     initial=np.r_[original.initial_coordinates(k,trial,cfg['seed']),obj.baseline_beta+phase];actual=initial.copy()
     for scale in [1.,.5,.25,.1,0.]:
      actual=np.r_[initial[:30]*scale,obj.baseline_beta+(initial[-1]-obj.baseline_beta)*scale]
      if obj.evaluate_b3(actual)[1].min()>=-1e-7:break
     opt=minimize(lambda z:obj.evaluate_b3(z)[0],actual,method='SLSQP',bounds=[(-12,12)]*24+[(-21,21)]*6+[(.005,.2)],constraints=[{'type':'ineq','fun':lambda z:obj.evaluate_b3(z)[1]}],options={'maxiter':cfg['max_iterations'],'ftol':2e-6,'eps':1e-4})
     obj.evaluate_b3(opt.x)
     saved=dict(context=context,trial=trial,round=round_no,seed=cfg['seed'],seed_entropy=[cfg['seed'],k,trial,3103],proposed=initial.tolist(),actual=actual.tolist(),success=bool(opt.success),stop_reason=str(opt.message),iterations=int(opt.nit),evaluations=obj.nfev,best=None if obj.best is None else dict(z=obj.best['z'].tolist(),distance=float(obj.best['distance']),R2_GD=float(obj.best['R2_GD']),R2_GX=float(obj.best['R2_GX'])),history=obj.history)
     write(checkpoint,dict(saved,payload_sha256=digest(saved)))
    evals+=saved['evaluations'];proposals.append(saved)
   trials_done+=size;screen=[];candidates=[];s=B3Search([r[::10] for r in bank_ref])
   for info in proposals:
    if info['best'] is None:continue
    z0=np.array(info['best']['z'])
    for scale in original.SCALES:
     z=np.r_[z0[:30]*scale,s.baseline_beta+(z0[-1]-s.baseline_beta)*scale];r,x=s.reconstruct_b3(z,original.DT);cv=(r-original.LOW)/(original.HIGH-original.LOW)
     feasible=bool(np.isfinite(r).all() and np.isfinite(x).all() and cv.min()>=0 and cv.max()<=1 and x[:,:10].min()>=0)
     d=min(original.distance(r,a,original.DT) for a in bank_ref) if feasible else None;db=original.distance(r,baseline,original.DT) if feasible else None
     row=dict(trial=info['trial'],scale=scale,beta=float(z[-1]),feasible=feasible,distance=d,baseline_distance=db,eligible=bool(feasible and d>.025 and db>.025));screen.append(row)
     if row['eligible']:candidates.append((d,r,z,info,scale,db,row))
   candidates.sort(key=lambda item:-item[0])
   for d,r,z,info,scale,db,row in candidates:
    verified,diagnostic=original.check_with_diagnostics(r,float(z[-1]),s.obs);row['acceptance']=diagnostic
    if verified is not None:accepted=(d,r,z,info,scale,db,verified);break
   write(root/'logs'/f'{member}_round{round_no}_screen.json',screen)
   rounds.append(dict(round=round_no,starts=size,candidates=len(screen),eligible=len(candidates)))
   if accepted is not None:break
  write(root/'logs'/(member+'_search_summary.json'),dict(rounds=rounds,starts=trials_done,evaluations=evals,accepted=accepted is not None))
  if accepted is None:raise RuntimeError(member+' failed after all allowed starts; checkpoints retained')
  d,r,z,info,scale,db,v=accepted
  meta={key:val for key,val in v.items() if key not in ['states','DOP853_daily_states','smoothness']};meta.update(solution=member,accepted=True,generation_index=k,selection_distance=d,baseline_distance=db,min_distance_to_prior_new=None if k==1 else d,search_starts_used=trials_done,search_rounds_used=len(rounds),total_search_evaluations=evals,selected_start=info['trial'],restoration_scale=scale,beta=float(z[-1]),pilot=args.pilot)
  np.savez_compressed(root/'solutions'/(member+'.npz'),rates=r,dt=original.DT,beta=float(z[-1]),states=v['states'],DOP853_daily_states=v['DOP853_daily_states'],coordinates=z)
  write(root/'solutions'/(member+'.json'),meta)
  import pandas as pd
  pd.DataFrame(v['smoothness']).to_csv(root/'solutions'/(member+'_smoothness.csv'),index=False)
  paths=[root/'solutions'/(member+suffix) for suffix in ['.npz','.json','_smoothness.csv']]
  write(marker,dict(files={str(p.relative_to(root)):sha(p) for p in paths},context=context));bank.append(r);manifest.append(meta)
  print('ACCEPTED',member,'beta',z[-1],'distance',d,flush=True)
 import pandas as pd
 matrix=np.zeros((len(bank),len(bank)))
 for i in range(len(bank)):
  for j in range(i):matrix[i,j]=matrix[j,i]=original.distance(bank[i],bank[j],original.DT)
 if len(bank)>1:
  for i,item in enumerate(manifest):item['min_distance_to_final_others']=float(np.delete(matrix[i],i).min())
 labels=[x['solution'] for x in manifest];pd.DataFrame(manifest).to_csv(root/'results/ensemble_manifest.csv',index=False);pd.DataFrame(matrix,index=labels,columns=labels).to_csv(root/'results/ensemble_distances.csv')
 files={}
 for name in labels:files.update(verify_marker(root,root/'checkpoints'/(name+'_accepted.json'))['files'])
 for rel in ['results/ensemble_manifest.csv','results/ensemble_distances.csv']:files[rel]=sha(root/rel)
 write(freeze,dict(base_path=str(root),member_ids=labels,n_solutions=len(labels),mode='fresh',files=files,frozen_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy_independent=True,source_code_hashes=code,pilot=args.pilot))
 protect('ensemble_freeze')
 print('FROZEN',len(labels),'pilot',args.pilot,flush=True)
if __name__=='__main__':main()
