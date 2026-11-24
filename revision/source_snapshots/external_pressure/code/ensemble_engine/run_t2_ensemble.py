"""T2 B1/B2 isolated original ensemble entrypoint; never calls old pipeline."""
from pathlib import Path
import json,os,sys,argparse,subprocess,hashlib
ENGINE=Path(__file__).resolve().parent;WORK=ENGINE.parents[2];PROJECT=WORK.parent
T2=WORK/'T2_ensemble_robustness';GROUPS=['B1_alt_baseline_A','B2_alt_baseline_B']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 a=argparse.ArgumentParser();a.add_argument('group',choices=GROUPS);a.add_argument('stage',choices=['pilot','pilot_current','prepare','generate','freeze','strategies']);args=a.parse_args()
 g=T2/args.group;pilot=args.stage in ['pilot','pilot_current'];out=g/'runs'/('engine_pilot' if args.stage=='pilot' else 'engine_pilot_current' if args.stage=='pilot_current' else 'ensemble')
 if not out.resolve().is_relative_to(WORK.resolve()):raise RuntimeError('Refuse output outside workspace')
 base=PROJECT/'baseline/Model4GDGXcq2x_v6_optimized01/Training-results-26-09-12_003513-set1' if args.stage=='pilot' else None
 if args.stage!='pilot':
  valid=g/'results/baseline_independent_validation.json'
  if not valid.is_file():raise RuntimeError('Formal baseline independent validation not yet completed')
  info=json.loads(valid.read_text());
  if not info['accepted']:raise RuntimeError('Formal baseline failed acceptance')
  base=Path(info['baseline_dir']).resolve()
  if not base.is_relative_to(WORK.resolve()):raise RuntimeError('Formal baseline outside workspace')
 ctx=out/'config_resolved.json'
 if args.stage in ['pilot','pilot_current','prepare']:
  if ctx.exists():raise RuntimeError('Context exists; refuse replacement')
  for name in ['results','figures','logs','solutions','methods','checkpoints']:(out/name).mkdir(parents=True,exist_ok=True)
  cfg=dict(project_root=str(PROJECT),output_dir=str(out),baseline_dir=str(base),source_analysis='uniform_30_solution_strategy_v2',risk_protocol='uniform_30_solution_strategy_v2/methods/fixed_risk_protocol.json',mode='fresh',n_solutions=1 if pilot else 30,seed=31415,n_starts=3,adaptive_start_rounds=[3,3,6],max_iterations=60,workers=3,nonmember_pilot=pilot)
  save(ctx,cfg)
  baseline_files=['config.json','beta.txt','gamma.txt','mu.txt','c1.txt','c2.txt','q1.txt','q2.txt','m1.txt','m2.txt','d1.txt','d2.txt']
  source_inputs={str(p):sha(p) for p in [*[base/name for name in baseline_files],PROJECT/'Data/input/Data_all_sm.xlsx',PROJECT/cfg['risk_protocol']]}
  save(out/'methods/source_hashes.json',dict(inputs=source_inputs,code={str(p.relative_to(ENGINE)):sha(p) for p in (ENGINE/'scripts').rglob('*.py')}))
  if not pilot:save(out/'results/baseline_audit.json',dict(passed=True,source=str(valid),validation_sha256=sha(valid),baseline=str(base)))
 if not ctx.is_file():raise RuntimeError('Run prepare first')
 hashes=json.loads((out/'methods/source_hashes.json').read_text(encoding='utf-8'))
 for p,h in hashes['inputs'].items():
  if not Path(p).is_file() or sha(p)!=h:raise RuntimeError('Baseline/risk/data source hash changed: '+p)
 for p,h in hashes['code'].items():
  if not (ENGINE/p).is_file() or sha(ENGINE/p)!=h:raise RuntimeError('Engine code hash changed: '+p)
 env=dict(os.environ,UDE_RUN_CONTEXT=str(ctx),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',UDE_FORBID_ODE='0',NUMBA_CACHE_DIR=str(g/'logs/numba_cache'),MPLCONFIGDIR=str(g/'logs/matplotlib'),TEMP=str(g/'logs/temp'),TMP=str(g/'logs/temp'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for key in ['NUMBA_CACHE_DIR','MPLCONFIGDIR','TEMP']:Path(env[key]).mkdir(parents=True,exist_ok=True)
 if args.stage=='prepare':print('Prepared',ctx);return
 if pilot:
  # Coarse candidate construction + free ODE evaluation. This intentionally
  # uses the frozen baseline only as a read-only reference, never as a member.
  cmd=[sys.executable,'-B','-c',"from scripts.generate_uniform_ensemble import *; obj=UniformSearch([np.zeros((T*4+1,8))]); z=initial_coordinates(1,0,SEED); trials=[]; chosen=None\nfor scale in [1.,.5,.25,.1,0.]:\n v,c,info=obj.evaluate(z*scale); trials.append(dict(scale=scale,objective=float(v),constraints=c.tolist(),details=info));\n if np.isfinite(c).all() and c.min()>=-1e-7: chosen=scale; break\natomic_json(ROOT/'results/nonmember_pilot.json',dict(nfev=obj.nfev,selected_scale=chosen,trials=trials,member=False));\nif chosen is None: raise RuntimeError('No feasible coarse pilot; do not launch formal search')\nprint('pilot candidate evaluated, coarse feasible; nonmember; nfev',obj.nfev)"]
 else:
  module={'generate':'scripts.generate_uniform_ensemble','freeze':'scripts.freeze_ensemble','strategies':'scripts.run_strategy_matrix'}[args.stage]
  if args.stage=='freeze':
   import pandas as pd
   m=out/'results/ensemble_manifest.csv'
   if not m.is_file() or len(pd.read_csv(m))!=30:raise RuntimeError('Ensemble must contain all 30 accepted solutions before freeze')
  if args.stage=='strategies' and not (out/'methods/ensemble_freeze.json').is_file():raise RuntimeError('Freeze before policy simulation')
  cmd=[sys.executable,'-B','-m',module]
 print('Running',args.stage,'with output',out,flush=True);subprocess.run(cmd,cwd=ENGINE,env=env,check=True)
if __name__=='__main__':main()
