"""Final, solver-free validation of numerical artifacts and protected inputs."""
from scripts.common import *
from scripts.analyze_pairwise_rankings import ranking_tables,FUNCTIONALS
from scripts.research_integrity import research_entries,changed_files,require_hashes,require_completed_analysis,verify_member_commits
from scripts.saved_artifact_audit import audit_policy,audit_members,audit_strategy_records
import datetime,re

def main():
    ctx=Context();original_protected=read_json(ctx.root/'methods/protected_before.json')
    protected=research_entries(original_protected);changed=changed_files(ctx.project,protected)
    state=read_json(ctx.root/'pipeline_state.json')
    require_completed_analysis(ctx.root,state)
    source=read_json(ctx.root/'methods/source_hashes.json')
    require_hashes(ctx.project,source['code'],'Current source code changed')
    require_hashes(ctx.project,{p:h for p,h in source['inputs'].items() if isinstance(h,str)},'Scientific input changed')
    # Read-only fingerprint calculation checks resolved config and runtime as well.
    from run_pipeline import fingerprints
    if fingerprints(ctx.config)[0]!=state['fingerprint']:raise RuntimeError('Run config/code/input/runtime fingerprint mismatch')
    if 'verify_only_upgrade' in state:
        entry=state['verify_only_upgrade'];event=ctx.root/entry['event']
        if sha(event)!=entry['event_sha256']:raise RuntimeError('Verify-only upgrade audit changed')
        require_hashes(ctx.project,read_json(event)['static_research_files'],'Maintained source/config changed')
    frozen=verify_ensemble(ctx);df=validate_scenarios(pd.read_csv(ctx.results/'complete_scenarios.csv'),ctx.n)
    policy_check=audit_policy(df,pd.read_csv(ctx.results/'policy_summary.csv'),ctx.n)
    member_check=audit_members(ctx.project,ctx.root,ctx.config,frozen,pd.read_csv(ctx.results/'ensemble_manifest.csv'))
    strategy_check=audit_strategy_records(ctx.root,ctx.n,read_json(ctx.protocol))
    expected,_,matrices=ranking_tables(df,ctx.n);actual=pd.read_csv(ctx.results/'pairwise_rankings.csv')
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-12,atol=1e-8)
    for name,mat in matrices.items():
        saved=pd.read_csv(ctx.results/f'pairwise_ranking_{name}.csv',index_col=0).values
        if not np.allclose(saved,mat,rtol=0,atol=1e-15) or not np.all(np.diag(saved)==0):raise AssertionError('Ranking matrix mismatch')
        for i,a in enumerate(STRATEGIES):
            for j,b in enumerate(STRATEGIES):
                row=actual[(actual.measure==name)&(actual.strategy_A==a)&(actual.strategy_B==b)].iloc[0]
                if row.A_fewer_count+row.B_fewer_count+row.equal_count!=ctx.n:raise AssertionError('Tie counts inconsistent')
    effects=pd.read_csv(ctx.results/'policy_effect_sizes.csv')
    for row in effects.itertuples():
        a,b,col=FUNCTIONALS[row.functional];d=df[(df.solution==row.solution)&(df.measure==row.measure)].set_index('strategy');value=d.loc[a,col]-(d.loc[b,col] if b else 0)
        if abs(value-row.effect_cases)>1e-7:raise AssertionError('Effect functional mismatch')
    m=pd.read_csv(ctx.results/'ensemble_manifest.csv')
    if m.solution.tolist()!=ids(ctx.n) or not m.accepted.all():raise AssertionError('Wrong member set')
    for col in ['R2_GD_RK4','R2_GX_RK4','R2_GD_DOP853','R2_GX_DOP853']:
        if not (m[col]>.98).all():raise AssertionError('Failed R2 acceptance')
    pair=pd.read_csv(ctx.results/'ensemble_distances.csv',index_col=0).values
    if pair.shape!=(ctx.n,ctx.n) or not np.allclose(pair,pair.T) or pair[np.triu_indices(ctx.n,1)].min()<=.025:raise AssertionError('Diversity contract failed')
    audit=pd.read_csv(ctx.results/'strategy_audit.csv')
    if len(audit)!=ctx.n*9 or not (audit.min_state>=0).all() or not (audit.mass_error<.01).all() or not (audit.RK_step_error<.05).all():raise AssertionError('Stored strategy solver audit failed')
    if not (audit[audit.strategy.isin(['HH','LL'])].DOP853_cumulative_error<.05).all():raise AssertionError('Stored DOP853 audit failed')
    if ctx.mode=='fresh':
        verify_member_commits(ctx.root,ctx.n)
        if len(list((ctx.root/'checkpoints').glob('*_accepted.json')))!=ctx.n:raise AssertionError('Missing accepted-member commits')
        if len(list((ctx.root/'checkpoints').glob('*_strategies.json')))!=ctx.n:raise AssertionError('Missing strategy commits')
        freeze_time=(ctx.root/'methods/ensemble_freeze.json').stat().st_mtime
        if any(p.stat().st_mtime<freeze_time for p in (ctx.results/'scenarios').glob('*_scenarios.csv')):raise AssertionError('Strategies written before freeze')
    doc=(ctx.root/'FINAL_REPORT.md').read_text(encoding='utf-8')
    for rel in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',doc):
        if not (ctx.root/rel).is_file():raise FileNotFoundError(rel)
    result=dict(passed=not changed,checked_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protected_files=len(protected),protected_manifest_entries=len(original_protected),excluded_nonresearch_paths=sorted(set(original_protected)-set(protected)),changed_files=changed,ensemble_frozen_unchanged=True,n_solutions=ctx.n,scenario_rows=len(df),strategy_scenarios=ctx.n*9,ranking_rows=len(actual),ranking_matrices=3,matrix_shape=[9,9],ranking_definition='strict fewer cases; ensemble fraction, not probability',effect_rows=len(effects),all_required_images_present=True,all_prior_stage_hashes_verified=True,member_and_strategy_commit_hashes_verified=ctx.mode=='fresh',final_report_sha256=sha(ctx.root/'FINAL_REPORT.md'),complete_scenarios_sha256=sha(ctx.results/'complete_scenarios.csv'),ODE_import_guard_active=os.environ.get('UDE_FORBID_ODE')=='1',ODE_recalculation_this_run=False,verification_scope='saved artifacts only; no scientific stage executed by this checker')
    result.update(independent_policy_audit=policy_check,saved_member_admissibility_audit=member_check,strategy_validation_scope=strategy_check)
    atomic_json(ctx.results/'integrity_check.json',result)
    if changed:raise RuntimeError('Protected files changed: '+str(changed))
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
