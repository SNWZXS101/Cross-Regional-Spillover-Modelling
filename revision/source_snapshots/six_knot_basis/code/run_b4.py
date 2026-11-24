"""B4 isolated checkpointable runner. No original pipeline or output directory.

Run stages in order; generation must finish before freeze, and freeze must
finish before any intervention. Existing accepted/scenario checkpoints are
verified and resumed by the copied stage implementations.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time

CODE=Path(__file__).resolve().parent; B4=CODE.parent;WORK=B4.parents[1];PROJECT=WORK.parent
OUT=B4/'runs/main';CONF=B4/'configs/main.json'
STAGES=['generate_uniform_ensemble','freeze_ensemble','run_strategy_matrix','analyze_pairwise_rankings','summarize_results','summarize_t2']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def validate():
 cfg=json.loads(CONF.read_text(encoding='utf-8'))
 for relative,expected in json.loads((B4/'configs/runner_sha256.json').read_text(encoding='utf-8')).items():
  if sha(B4/relative)!=expected:raise RuntimeError('Runner/config changed since registration: '+relative)
 if Path(cfg['output_dir']).resolve()!=OUT.resolve() or not OUT.resolve().is_relative_to(WORK.resolve()):raise RuntimeError('B4 output must be inside this revision workspace')
 if Path(cfg['project_root']).resolve()!=PROJECT.resolve():raise RuntimeError('Unexpected scientific source root')
 if cfg['mode']!='fresh' or cfg['n_solutions']!=30 or cfg['n_starts']!=3 or cfg['max_iterations']!=60 or cfg['adaptive_start_rounds']!=[3,3,6]:raise RuntimeError('Formal B4 budget or mode changed')
 for rel,h in json.loads((B4/'configs/input_sha256.json').read_text(encoding='utf-8')).items():
  path=PROJECT/rel
  if not path.is_file() or sha(path)!=h:raise RuntimeError('Frozen input changed: '+str(path))
 for rel,item in json.loads((B4/'configs/code_sha256.json').read_text(encoding='utf-8')).items():
  if sha(CODE/rel)!=item['sha256']:raise RuntimeError('Copied code changed after registration: '+rel)
 source=json.loads((OUT/'methods/source_hashes.json').read_text(encoding='utf-8'))
 if source['inputs']!=json.loads((B4/'configs/input_sha256.json').read_text(encoding='utf-8')) or source['code']!=json.loads((B4/'configs/code_sha256.json').read_text(encoding='utf-8')):raise RuntimeError('Run provenance does not match registered source and code')
 freeze=OUT/'methods/ensemble_freeze.json'
 if freeze.exists() and json.loads(freeze.read_text(encoding='utf-8'))['source_code_hashes']!=source['code']:raise RuntimeError('Frozen ensemble was generated with a different code version')
 return cfg
def verify_original_sources(stage,env,stream):
    verifier=WORK/'T2_ensemble_robustness/verify_original_sources.py'
    result=subprocess.run([sys.executable,'-B',str(verifier),'--label','B4_'+stage],cwd=WORK,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Original scientific source protection failed; stop B4')
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES+['all','check'],default='check');a=p.parse_args()
 cfg=validate()
 if a.stage=='check':print('B4 paths, budget, frozen inputs and copied code hashes verified; no calculation started.');return
 env=dict(os.environ,UDE_RUN_CONTEXT=str(CONF),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 cache=OUT/'cache';tmp=OUT/'tmp';cache.mkdir(parents=True,exist_ok=True);tmp.mkdir(parents=True,exist_ok=True)
 env.update(NUMBA_CACHE_DIR=str(cache),TEMP=str(tmp),TMP=str(tmp))
 stages=STAGES if a.stage=='all' else [a.stage]
 for stage in stages:
  freeze=OUT/'methods/ensemble_freeze.json'
  if stage=='generate_uniform_ensemble' and freeze.exists():
   print('Already frozen: generation skipped; following stages verify freeze.',flush=True);continue
  if stage in STAGES[2:] and not freeze.is_file():raise RuntimeError('Cannot run policy/statistics before complete ensemble freeze')
  validate();log=B4/'logs'/('main_'+stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('a',encoding='utf-8') as stream:
   stream.write('\nB4 stage '+stage+' started '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n');stream.flush()
   result=subprocess.run([sys.executable,'-B','-m','scripts.'+stage],cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT)
   verify_original_sources(stage,env,stream)
  validate()
  if result.returncode:raise RuntimeError('B4 '+stage+' failed; verified completed checkpoints remain; see '+str(log))
  if stage=='summarize_t2':
   import pandas as pd
   from scripts.common import validate_scenarios,ids,STRATEGIES
   import itertools,numpy as np
   freeze=json.loads((OUT/'methods/ensemble_freeze.json').read_text(encoding='utf-8'))
   scenarios=pd.read_csv(OUT/'results/complete_scenarios.csv');audits=pd.read_csv(OUT/'results/strategy_audit.csv')
   rankings=pd.read_csv(OUT/'results/T2_pairwise_counts.csv');effects=pd.read_csv(OUT/'results/T2_core_effects.csv');per_solution=pd.read_csv(OUT/'results/T2_policy_effects.csv')
   if freeze['n_solutions']!=30 or freeze['member_ids']!=ids(30) or freeze.get('policy_independent') is not True or Path(freeze['base_path']).resolve()!=OUT.resolve():raise RuntimeError('B4 freeze membership/provenance invalid')
   for rel,expected in freeze['files'].items():
    if sha(Path(freeze['base_path'])/rel)!=expected:raise RuntimeError('B4 frozen member artifact changed: '+rel)
   normalized={str(rel).replace('\\','/'):h for rel,h in freeze['files'].items()}
   for name in ids(30):
    for ext in ['npz','json']:
     if 'solutions/'+name+'.'+ext not in normalized:raise RuntimeError('B4 frozen member missing: '+name+'.'+ext)
   validate_scenarios(scenarios,30)
   expected_audits=set(itertools.product(ids(30),STRATEGIES))
   if len(audits)!=270 or audits.duplicated(['solution','strategy']).any() or set(zip(audits.solution,audits.strategy))!=expected_audits:raise RuntimeError('Incomplete B4 strategy audit')
   if not np.isfinite(audits.RK_step_error).all() or not (audits.RK_step_error<.05).all():raise RuntimeError('B4 saved strategy RK4 consistency fails')
   expected_pairs={(m,a,b) for m in ['reported','all_infections'] for a,b in itertools.combinations(STRATEGIES,2)}
   if len(rankings)!=72 or rankings.duplicated(['measure','strategy_A','strategy_B']).any() or set(zip(rankings.measure,rankings.strategy_A,rankings.strategy_B))!=expected_pairs:raise RuntimeError('Incomplete B4 pairwise rankings')
   expected_effects={'A_GD_HN','A_GX_NH','G_HH_minus_LL','A_total_HH'}
   expected_effect_rows=set(itertools.product(['reported','all_infections'],expected_effects))
   expected_member_rows=set(itertools.product(ids(30),['reported','all_infections'],expected_effects))
   if len(effects)!=8 or effects.duplicated(['measure','functional']).any() or set(zip(effects.measure,effects.functional))!=expected_effect_rows or len(per_solution)!=240 or per_solution.duplicated(['solution','measure','functional']).any() or set(zip(per_solution.solution,per_solution.measure,per_solution.functional))!=expected_member_rows:raise RuntimeError('Incomplete B4 shared effect statistics')
   integrity=WORK/'T2_ensemble_robustness/logs/original_integrity_B4_summarize_t2.json'
   if json.loads(integrity.read_text(encoding='utf-8')).get('passed') is not True:raise RuntimeError('B4 original scientific source protection failed')
   marker={'completed':True,'n_solutions':30,'policy_selection_blind':True,'freeze_sha256':sha(OUT/'methods/ensemble_freeze.json'),
    'scenario_sha256':sha(OUT/'results/complete_scenarios.csv'),'strategy_audit_sha256':sha(OUT/'results/strategy_audit.csv'),
    'policy_summary_sha256':sha(OUT/'results/T2_core_effects.csv'),'policy_ranking_sha256':sha(OUT/'results/T2_pairwise_counts.csv'),
    'baseline_validation_sha256':sha(OUT/'results/baseline_audit.json'),
    'original_integrity_sha256':sha(integrity)}
   target=B4/'reports/T2_group_complete.json';target.parent.mkdir(parents=True,exist_ok=True)
   target.write_text(json.dumps(marker,indent=2)+'\n',encoding='utf-8')
  print('B4 completed',stage,'log:',log,flush=True)
if __name__=='__main__':main()
