"""Run a new policy-blind ensemble and nine strategies in a new output folder.
This long-running computation never writes to the archived results.
"""
from pathlib import Path
import os,sys,json,argparse,hashlib,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output',type=Path,required=True,help='new output directory, not an archived results folder')
ap.add_argument('--n-solutions',type=int,default=100);ap.add_argument('--seed',type=int,default=31415)
ap.add_argument('--n-starts',type=int,default=3);ap.add_argument('--max-iterations',type=int,default=60);ap.add_argument('--workers',type=int,default=1)
ap.add_argument('--through',choices=['audit_baseline','generate_uniform_ensemble','freeze_ensemble','run_strategy_matrix'],default='run_strategy_matrix')
a=ap.parse_args();out=a.output.resolve();archive=(ROOT/'results').resolve()
if out.exists() or any(out==p or p in out.parents for p in [archive,(ROOT/'data').resolve(),(ROOT/'model').resolve(),HERE.resolve()]):raise SystemExit('Output must be a new directory outside data/, model/, analysis/, and results/.')
if min(a.n_solutions,a.n_starts,a.max_iterations,a.workers)<1:raise SystemExit('Positive counts required')
if a.n_solutions<2:raise SystemExit('At least two solutions are required for distance diagnostics')
out.mkdir(parents=True)
for d in ['results','figures','logs','solutions','methods','checkpoints']:(out/d).mkdir()
cfg=dict(project_root=str(ROOT),output_dir=str(out),mode='fresh',n_solutions=a.n_solutions,seed=a.seed,n_starts=a.n_starts,adaptive_start_rounds=[a.n_starts,a.n_starts,2*a.n_starts],max_iterations=a.max_iterations,workers=a.workers,baseline_dir='data/baseline',risk_protocol='analysis/protocol/fixed_risk_protocol.json',source_analysis='results/sensitivity_100')
context=out/'config_resolved.json';context.write_text(json.dumps(cfg,indent=2),encoding='utf8')
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('scripts/**/*.py')}
input_files=list((ROOT/'data/baseline').glob('*.txt'))+[ROOT/'data/baseline/config.json',ROOT/'data/input/Data_all_sm.xlsx',HERE/'protocol/fixed_risk_protocol.json']
inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}
(out/'methods/source_hashes.json').write_text(json.dumps({'code':hashes,'inputs':inputs},indent=2),encoding='utf8')
env=os.environ.copy();env.update(UDE_RUN_CONTEXT=str(context),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(HERE)+os.pathsep+env.get('PYTHONPATH',''))
for stage in ['audit_baseline','generate_uniform_ensemble','freeze_ensemble','run_strategy_matrix']:
 print('Running',stage,flush=True)
 with (out/'logs'/f'{stage}.log').open('w',encoding='utf8') as log:
  result=subprocess.run([sys.executable,'-B','-m','scripts.'+stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise SystemExit(f'{stage} failed; see {out}/logs/{stage}.log')
 if stage==a.through:break
