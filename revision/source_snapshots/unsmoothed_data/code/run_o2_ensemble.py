"""O2 policy-blind 30-member ensemble pipeline; accepts only independently verified baseline."""
from pathlib import Path
import sys,os,json,hashlib,subprocess
CODE=Path(__file__).resolve().parent;GROUP=CODE.parent;T3=GROUP.parent;WORK=T3.parent;PROJECT=WORK.parent;ENGINE=CODE/'engine';OUT=GROUP/'runs/ensemble'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 global OUT
 if len(sys.argv)!=2 or sys.argv[1] not in ('pilot','prepare','generate','freeze','strategies'):raise ValueError('Specify pilot, prepare, generate, freeze or strategies')
 stage=sys.argv[1]; valid=GROUP/'results/baseline_independent_validation.json'
 if stage=='pilot':OUT=GROUP/'runs/accepted_baseline_pilot'
 if not valid.is_file():raise RuntimeError('O2 independent baseline acceptance missing')
 info=json.loads(valid.read_text(encoding='utf-8'));base=Path(info['baseline_dir']).resolve()
 if not info['accepted'] or not base.is_relative_to((GROUP/'runs/baseline_training').resolve()):raise RuntimeError('Baseline not accepted or path invalid')
 cfgpath=OUT/'config_resolved.json';proofpath=OUT/'methods/source_hashes.json'
 if stage in ('prepare','pilot'):
  if cfgpath.exists():raise RuntimeError('Existing context; do not overwrite')
  for name in ('results','figures','logs','solutions','methods','checkpoints'):(OUT/name).mkdir(parents=True,exist_ok=True)
  cfg=dict(project_root=str(PROJECT),output_dir=str(OUT),baseline_dir=str(base),source_analysis='uniform_30_solution_strategy_v2',risk_protocol='uniform_30_solution_strategy_v2/methods/fixed_risk_protocol.json',mode='fresh',n_solutions=1 if stage=='pilot' else 30,seed=31415,n_starts=3,adaptive_start_rounds=[3,3,6],max_iterations=60,workers=3,nonmember_pilot=stage=='pilot')
  cfgpath.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
  inputs=[base/k for k in ['config.json','beta.txt','gamma.txt','mu.txt','c1.txt','c2.txt','q1.txt','q2.txt','m1.txt','m2.txt','d1.txt','d2.txt']]+[PROJECT/'Data/input/Data_all.xlsx',PROJECT/cfg['risk_protocol'],valid,GROUP/'configs/admissibility_prepolicy.json']
  proof=dict(inputs={str(p):sha(p) for p in inputs},code={str(p.relative_to(ENGINE)):sha(p) for p in (ENGINE/'scripts').rglob('*.py')},runners={str(p):sha(p) for p in [CODE/'run_o2_ensemble.py',CODE/'train_o2.py',CODE/'ude_training.py',CODE/'validate_o2.py',CODE/'independent_validator.py',CODE/'fit_diagnostics.py',CODE/'o2_queue.py',WORK/'shared/code/summarize_policy.py']})
  proofpath.write_text(json.dumps(proof,indent=2),encoding='utf-8');(OUT/'results/baseline_audit.json').write_text(json.dumps({'passed':True,'source':str(valid),'baseline':str(base)},indent=2),encoding='utf-8')
 if not cfgpath.is_file() or not proofpath.is_file():raise RuntimeError('O2 prepare required')
 proof=json.loads(proofpath.read_text(encoding='utf-8'))
 for p,h in proof['inputs'].items():
  if sha(Path(p))!=h:raise RuntimeError('Input changed: '+p)
 for p,h in proof['code'].items():
  if sha(ENGINE/p)!=h:raise RuntimeError('Code changed: '+p)
 for p,h in proof['runners'].items():
  if sha(Path(p))!=h:raise RuntimeError('Runner changed: '+p)
 if stage=='prepare':print('O2 context prepared',flush=True);return
 if stage=='generate' and (OUT/'methods/ensemble_freeze.json').exists():raise RuntimeError('Frozen ensemble; generation forbidden')
 if stage=='strategies' and not (OUT/'methods/ensemble_freeze.json').exists():raise RuntimeError('Ensemble must be frozen')
 env=dict(os.environ,UDE_RUN_CONTEXT=str(cfgpath),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TEMP=str(GROUP/'logs/runtime/temp'),TMP=str(GROUP/'logs/runtime/temp'),NUMBA_CACHE_DIR=str(GROUP/'logs/runtime/numba'),MPLCONFIGDIR=str(GROUP/'logs/runtime/matplotlib'))
 if stage=='pilot':
  snippet="from scripts.generate_uniform_ensemble import *\no=UniformSearch([np.zeros((T*4+1,8))]); z=initial_coordinates(1,0,SEED); chosen=None\nfor scale in [1.,.5,.25,.1,0.]:\n v,c,info=o.evaluate(z*scale)\n if np.isfinite(c).all() and c.min()>=-1e-7: chosen=z*scale; break\nif chosen is None: raise RuntimeError('No coarse feasible pilot')\nr,h=o.construct(chosen,DT); verified,diag=check_with_diagnostics(r,o.beta,o.obs); tt=grid(.025); x,mn,mass=simulate(r,DT,o.beta,tt,factors(tt,('N','N')),.00625)\natomic_json(ROOT/'results/nonmember_pilot.json',dict(finite=bool(np.isfinite(x).all()),min_state=float(mn),mass_error=float(mass),nfev=o.nfev,independent_acceptance=diag,member=False))\nif verified is None or not np.isfinite(x).all() or mn<0 or mass>=.01: raise RuntimeError('Pilot fails independent validation')"
  subprocess.run([sys.executable,'-B','-c',snippet],cwd=ENGINE,env=env,check=True)
 else:
  module={'generate':'generate_uniform_ensemble','freeze':'freeze_ensemble','strategies':'run_strategy_matrix'}[stage]
  subprocess.run([sys.executable,'-B','-m','scripts.'+module],cwd=ENGINE,env=env,check=True)
if __name__=='__main__':main()



