"""M2 ensemble stages; all policy blind until freeze; no old outputs."""
from pathlib import Path
import sys,os,json,hashlib,subprocess,argparse
M2=Path(__file__).resolve().parent.parent;T3=M2.parent;WORK=T3.parent;PROJECT=WORK.parent;ENGINE=M2/'code/ensemble_engine'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise RuntimeError('Refusing overwrite '+str(p))
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','prepare','generate','freeze','strategies']);a=ap.parse_args()
 info=json.loads((M2/'results/baseline_independent_validation.json').read_text(encoding='utf-8'))
 if not info.get('accepted'):raise RuntimeError('M2 baseline did not pass validation')
 base=Path(info['baseline_dir']).resolve()
 if not base.is_relative_to(M2.resolve()):raise RuntimeError('Refusing baseline outside M2')
 for p,h in info['source_sha256'].items():
  if sha(p)!=h:raise RuntimeError('M2 baseline input changed: '+p)
 pilot=a.stage=='pilot';out=M2/'runs'/('engine_pilot' if pilot else 'ensemble');ctx=out/'config_resolved.json'
 if a.stage in ('pilot','prepare'):
  if ctx.exists():raise RuntimeError('Output context exists; do not overwrite')
  for name in ('results','figures','logs','solutions','methods','checkpoints'):(out/name).mkdir(parents=True,exist_ok=True)
  cfg=dict(project_root=str(PROJECT),output_dir=str(out),baseline_dir=str(base),source_analysis='uniform_30_solution_strategy_v2',risk_protocol='uniform_30_solution_strategy_v2/methods/fixed_risk_protocol.json',mode='fresh',n_solutions=1 if pilot else 30,seed=31415,n_starts=3,adaptive_start_rounds=[3,3,6],max_iterations=60,workers=3,nonmember_pilot=pilot)
  save(ctx,cfg)
  paths=[M2/'configs/calibration.json',M2/'results/baseline_independent_validation.json',PROJECT/'Data/input/Data_all_sm.xlsx',PROJECT/cfg['risk_protocol'],*[base/(n+'.txt') for n in ['c1','c2','q1','q2','m1','m2','d1','d2','beta','gamma','mu']],base/'config.json']
  save(out/'methods/source_hashes.json',{'inputs':{str(p):sha(p) for p in paths},'code':{str(p.relative_to(ENGINE)):sha(p) for p in ENGINE.rglob('*.py')},'policy_blind':True})
 if not ctx.is_file():raise RuntimeError('Prepare context first')
 hashes=json.loads((out/'methods/source_hashes.json').read_text())
 for p,h in hashes['inputs'].items():
  if sha(p)!=h:raise RuntimeError('Source changed '+p)
 for p,h in hashes['code'].items():
  if sha(ENGINE/p)!=h:raise RuntimeError('Code changed '+p)
 env=dict(os.environ,UDE_RUN_CONTEXT=str(ctx),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(M2/'logs/numba_cache'),TEMP=str(M2/'logs/temp'),TMP=str(M2/'logs/temp'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 if a.stage=='prepare':print('Prepared',ctx);return
 if a.stage=='pilot':
  script="""from scripts.generate_uniform_ensemble import *\ns=UniformSearch([np.zeros((T*4+1,8))]);r,x=s.construct(np.zeros(30),DT)\nif not np.isfinite(r).all() or not np.isfinite(x).all():raise RuntimeError('M2 zero deformation reconstruction infeasible')\nprotected=np.arange(len(r))*DT<=7\nif not np.allclose(r[protected,2:],s.reference[protected,2:],rtol=0,atol=1e-14):raise RuntimeError('Initial protection mismatch')\nv,detail=check_with_diagnostics(r,s.beta,s.obs)\nif v is None:raise RuntimeError('M2 zero deformation independent acceptance failed: '+str(detail))\natomic_json(ROOT/'results/nonmember_pilot.json',dict(passed=True,member=False,detail=detail,beta=s.beta,maximum_rate_difference=float(np.max(np.abs(r-s.reference)))))\nprint('M2 zero deformation independent acceptance passed; not a member',flush=True)"""
  cmd=[sys.executable,'-B','-c',script]
 else:
  module={'generate':'scripts.generate_uniform_ensemble','freeze':'scripts.freeze_ensemble','strategies':'scripts.run_strategy_matrix'}[a.stage]
  if a.stage=='freeze':
   import pandas as pd
   m=out/'results/ensemble_manifest.csv'
   if not m.is_file() or len(pd.read_csv(m))!=30:raise RuntimeError('Need all 30 accepted before freeze')
  if a.stage=='strategies' and not (out/'methods/ensemble_freeze.json').is_file():raise RuntimeError('Freeze before strategies')
  cmd=[sys.executable,'-B','-m',module]
 print('M2 stage',a.stage,'output',out,flush=True);subprocess.run(cmd,cwd=ENGINE,env=env,check=True)
if __name__=='__main__':main()
