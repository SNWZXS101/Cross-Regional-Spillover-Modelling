"""Read-only manuscript-era audit; outputs only to a new maintenance directory."""
import os,sys,argparse
os.environ['UDE_FORBID_ODE']='1'
from scripts import _NoODEImports
sys.meta_path.insert(0,_NoODEImports())
from scripts.common import Path,atomic_json,sha
from scripts.saved_artifact_audit import audit_run

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,default=Path('maintenance/manuscript_architecture_20260915/audits'))
    parser.add_argument('--runs',nargs='+',choices=['main_30','sensitivity_50','sensitivity_100'],default=['main_30','sensitivity_50','sensitivity_100'])
    args=parser.parse_args();project=args.project.resolve();output=(project/args.output).resolve()
    if not output.is_relative_to(project/'maintenance'):raise ValueError('Audit writes only to maintenance/, never frozen analysis_runs')
    for name in args.runs:
        print('Read-only saved-artifact audit:',name,flush=True)
        try:result=audit_run(project,name)
        except Exception as error:
            atomic_json(output/(name+'_FAILED.json'),dict(passed=False,run=name,error=str(error),ODE_calls=0));raise
        result['audit_code_sha256']={rel:sha(project/rel) for rel in ['scripts/audit_manuscript.py','scripts/saved_artifact_audit.py','scripts/common.py','scripts/research_integrity.py']}
        result['python_optimization_level']=sys.flags.optimize
        atomic_json(output/(name+'.json'),result)
        print('PASS:',name,'members',result['n'],'pairs',result['members']['pairs_recomputed'],flush=True)
    atomic_json(output/'summary.json',dict(passed=True,runs=args.runs,ODE_calls=0,read_only_scientific_runs=True))

if __name__=='__main__':main()
