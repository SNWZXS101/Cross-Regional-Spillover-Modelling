"""Summarize completed T5 run and bind outputs to source/checkpoint hashes."""
from pathlib import Path
import argparse,hashlib,json,datetime
import numpy as np,pandas as pd
T5=Path(__file__).resolve().parents[2];SOURCE=T5.parent.parent/'analysis_runs/sensitivity_100'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(c,m):
 if not c:raise RuntimeError(m)
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',required=True);a=p.parse_args();root=(T5/a.group).resolve();require(root.is_relative_to(T5.resolve()),'outside T5')
 ids=[f'U{i:02d}' for i in range(1,101)];sc=root/'results/complete_scenarios.csv';au=root/'results/strategy_audit.csv';m=pd.read_csv(sc);d=pd.read_csv(au)
 require(len(m)==1800 and len(d)==900 and set(m.solution)==set(ids) and set(d.solution)==set(ids),'wrong counts')
 require(not m.duplicated(['solution','strategy','measure']).any() and not d.duplicated(['solution','strategy']).any(),'duplicate scenarios')
 for ident in ids:
  obj=json.loads((root/'checkpoints'/f'{ident}_strategies.json').read_text());require(sha(SOURCE/'solutions'/f'{ident}.npz')==obj['source_sha256'],'source mismatch')
  for rel,h in obj['files'].items():require(sha(root/rel)==h,'checkpoint mismatch')
 prov=json.loads((root/'methods/source.json').read_text());ctx=json.loads((root/'methods/run_context.json').read_text())
 require(sha(ctx['risk_protocol'])==prov['protocol_sha256'],'protocol mismatch')
 for key,path in [('runner_sha256',T5/'shared/code/run_group.py'),('solver_sha256',T5/'shared/code/scripts/core/solvers.py'),('model_sha256',T5/'shared/code/scripts/core/model.py'),('independent_sha256',T5/'shared/code/scripts/core/independent_rhs.py')]:require(sha(path)==prov[key],'code mismatch')
 log=root/'logs/formal_stdout.log';start=datetime.datetime.fromtimestamp(log.stat().st_ctime,datetime.timezone.utc);end=datetime.datetime.fromtimestamp(sc.stat().st_mtime,datetime.timezone.utc)
 report={'group':a.group,'n':100,'scenario_rows':len(m),'audit_rows':len(d),'all_checkpoints_passed':True,'start_utc':start.isoformat(),'completed_utc':end.isoformat(),'elapsed_seconds':(end-start).total_seconds(),'maximum_RK_error':float(d.RK_step_error.max()),'maximum_DOP_error':float(d.DOP853_cumulative_error.max()),'fallback_count':int((d.DOP853_method=='DOP853 each rate interval').sum()),'maximum_initial_event_error':float(d.DOP853_event_cumulative_error.max()),'scenario_sha256':sha(sc),'audit_sha256':sha(au),'source_sha256':sha(root/'methods/source.json'),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 dest=root/'results/run_certificate.json';require(not dest.exists(),'refusing overwrite');dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
