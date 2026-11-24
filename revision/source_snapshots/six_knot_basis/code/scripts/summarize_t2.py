"""Run the shared T2 saved-scenario summary with B4 frozen inputs only."""
from scripts.common import *
import importlib.util
from pathlib import Path
WORK=Path(__file__).resolve().parents[4]
shared=WORK/'shared/code/summarize_policy.py'
spec=importlib.util.spec_from_file_location('t2_shared_policy',shared)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def main():
 ctx=Context();verify_ensemble(ctx)
 scenarios=pd.read_csv(ctx.results/'complete_scenarios.csv');audits=pd.read_csv(ctx.results/'strategy_audit.csv')
 validate_scenarios(scenarios,ctx.n)
 rankings,effects,summary=module.analyze(scenarios,audits)
 rankings.to_csv(ctx.results/'T2_pairwise_counts.csv',index=False)
 effects.to_csv(ctx.results/'T2_policy_effects.csv',index=False,float_format='%.17g')
 summary.to_csv(ctx.results/'T2_core_effects.csv',index=False,float_format='%.17g')
 atomic_json(ctx.results/'T2_resolution_note.json',dict(near_zero_rule='Shared T2 indicator from saved fine/coarse RK differences; unresolved requires targeted finer checks; not a rigorous error bound.',policy_blind_ensemble=True,n=ctx.n))
 print('Shared T2 summary saved: two outcomes, all pairs, four functionals; near-zero unresolved.')
if __name__=='__main__':main()
