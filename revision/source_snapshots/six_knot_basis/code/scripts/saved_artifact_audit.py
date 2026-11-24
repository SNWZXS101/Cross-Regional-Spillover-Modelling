"""Independent algebraic audit of frozen artifacts. No core/solver imports.

This module reconstructs classifications, summaries and distances from saved
numbers. Historical finer-grid error estimates are checked as recorded evidence;
unretained RK4 substeps cannot be independently recovered without integration.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scripts.common import read_json, ids, sha, STRATEGIES, validate_scenarios
from scripts.research_integrity import require_completed_analysis, require_hashes, verify_member_commits

RATES=['c1','c2','q1','q2','m1','m2','d1','d2']
LABELS=['win-win','self-harm / neighbor-benefit','burden shifting','lose-lose','neutral','mixed with neutral']

def require(condition,message):
    if not condition:raise RuntimeError(message)

def close(actual,expected,label,atol=1e-8,rtol=1e-12):
    require(np.allclose(actual,expected,atol=atol,rtol=rtol,equal_nan=False),label+' mismatch')

def classify_saved(a1,a2,strategy,tolerance=.01):
    """Independent scalar truth table; unilateral controller is the non-N origin."""
    require(strategy in STRATEGIES,'Unknown strategy: '+str(strategy))
    require(np.isfinite([a1,a2]).all(),'Nonfinite effect in classification')
    signs=[1 if x>tolerance else -1 if x< -tolerance else 0 for x in [a1,a2]]
    if signs==[1,1]:return 'win-win'
    if signs==[-1,-1]:return 'lose-lose'
    if signs==[0,0]:return 'neutral'
    if 0 in signs:return 'mixed with neutral'
    controller=0 if strategy[0]!='N' and strategy[1]=='N' else 1 if strategy[0]=='N' and strategy[1]!='N' else None
    if controller is not None and signs[controller]==-1:return 'self-harm / neighbor-benefit'
    return 'burden shifting'

def reconstruct_policy(df,n):
    """Rebuild every summary column without consuming stored classifications."""
    frame=validate_scenarios(df.copy(),n)
    frame['classification']=[classify_saved(r.A1,r.A2,r.strategy) for r in frame.itertuples()]
    rows=[]
    for measure in sorted(frame.measure.unique()):
        for strategy in sorted(STRATEGIES):
            part=frame[(frame.measure==measure)&(frame.strategy==strategy)]
            row={'measure':measure,'strategy':strategy,'n':n}
            labels=part.classification.tolist()
            for label in LABELS:
                prefix=label.replace(' / ','_').replace('-','_').replace(' ','_')
                count=labels.count(label);row[prefix+'_count']=count;row[prefix+'_fraction']=count/n
            for column in ['A1','A2','Atotal']:
                values=part[column].to_numpy()
                row[column+'_positive_count']=int(np.count_nonzero(values>.01))
                row[column+'_negative_count']=int(np.count_nonzero(values<-.01))
                for key,value in [('min',np.min(values)),('median',np.median(values)),('max',np.max(values))]:row[column+'_'+key]=float(value)
            rows.append(row)
    return frame,pd.DataFrame(rows)

def audit_policy(df,summary,n):
    require('classification' in df,'Missing saved classification')
    calculated,expected=reconstruct_policy(df,n)
    mismatch=df.classification.to_numpy()!=calculated.classification.to_numpy()
    require(not mismatch.any(),'Saved outcome classification differs: '+str(df.loc[mismatch,['solution','strategy','measure']].head().to_dict('records')))
    if 'win_win' in df:require(np.array_equal(df.win_win.to_numpy(),calculated.classification.eq('win-win').to_numpy()),'Saved win_win flag differs')
    for field,i in [('GD_strategy',0),('GX_strategy',1)]:
        if field in df:require(df[field].eq(df.strategy.str[i]).all(),field+' inconsistent with strategy')
    require(set(summary)==set(expected),'policy_summary column contract differs')
    actual=summary.sort_values(['measure','strategy']).reset_index(drop=True)[expected.columns]
    require(len(actual)==18,'policy_summary must contain 18 rows')
    for column in expected:
        if column in ['measure','strategy']:require(actual[column].equals(expected[column]),column+' summary keys differ')
        elif column=='n' or column.endswith('_count'):require(np.array_equal(actual[column],expected[column]),column+' summary counts differ')
        else:close(actual[column],expected[column],column+' summary')
    return dict(classification_rows=len(df),policy_summary_rows=18,policy_summary_columns=len(expected.columns),
                independently_recomputed=True,summary_records=expected.to_dict('records'))

def functional_distance(a,b,low,high,dt,horizon):
    # Exact integral of the square of a linearly interpolated rate difference.
    z=(a-b)/(high-low)
    integrals=((z[:-1]**2+z[:-1]*z[1:]+z[1:]**2)*(dt/3)).sum(axis=0)
    return float(np.sqrt(integrals/horizon).mean())

def boundary_fraction(values,threshold=.02):
    left=np.minimum(values[:-1],values[1:]);right=np.maximum(values[:-1],values[1:])
    fractions=np.zeros_like(left);fractions[right<=threshold]=1
    crossing=(left<threshold)&(right>threshold)
    fractions[crossing]=(threshold-left[crossing])/(right[crossing]-left[crossing])
    return float(fractions.mean())

def validate_distance_matrix(matrix,n,threshold=.025):
    a=np.asarray(matrix)
    require(a.shape==(n,n) and np.isfinite(a).all(),'Invalid distance matrix shape/finiteness')
    require(np.array_equal(a,a.T),'Distance matrix not symmetric')
    require(np.array_equal(np.diag(a),np.zeros(n)),'Distance diagonal not exactly zero')
    require((a[np.triu_indices(n,1)]>threshold).all(),'Final pairwise distance fails strict novelty threshold')

def state_checks(states,shape,initial,total_population,label):
    require(states.shape==shape,label+' shape differs')
    require(np.isfinite(states).all() and (states>=0).all(),label+' nonfinite or negative saved state')
    close(states[0,:12],initial,label+' fixed initial state',atol=0,rtol=0)
    mass=float(np.abs(states[:,:10].sum(axis=1)-total_population).max())
    require(mass<=.01,label+' total population not conserved')
    return mass

def audit_members(project,run,config,freeze,manifest):
    base=Path(freeze['base_path']);n=int(config['n_solutions']);expected_ids=ids(n)
    require(freeze['member_ids']==expected_ids and freeze['n_solutions']==n,'Freeze member IDs/order differ')
    require(manifest.solution.tolist()==expected_ids,'Manifest member IDs/order differ')
    require(manifest.accepted.eq(True).all(),'Manifest contains unaccepted members')
    actual={p.stem for p in (base/'solutions').glob('U*.npz')}
    require(actual==set(expected_ids),'Frozen member file set differs')
    baseline=project/config['baseline_dir'];cfg=read_json(baseline/'config.json')
    protocol=read_json(project/config['risk_protocol']);horizon=int(protocol['horizon'])
    low=np.array([cfg[k+'_bounds'][0] for k in RATES]);high=np.array([cfg[k+'_bounds'][1] for k in RATES])
    beta=float(np.loadtxt(baseline/'beta.txt'));pop=np.array([cfg['N1_raw'],cfg['N2_raw']])
    require(np.isfinite(beta) and cfg['beta_bounds'][0]<=beta<=cfg['beta_bounds'][1],'Frozen baseline beta out of bounds')
    initial=np.array([0.,0.]+[cfg[k+'0_raw'] for k in ['Sq1','Sq2','I1','I2','Iq1','Iq2','R1','R2']]+[0.,0.])
    initial[:2]=pop-initial[2:10].reshape(4,2).sum(axis=0)
    with np.load(run/'results/baseline_rollout.npz',allow_pickle=False) as b:observed=b['observed'].copy()
    require(observed.shape==(horizon,2) and np.isfinite(observed).all(),'Invalid saved observations')
    dist_frame=pd.read_csv(run/'results/ensemble_distances.csv',index_col=0)
    require(dist_frame.index.tolist()==expected_ids and dist_frame.columns.tolist()==expected_ids,'Distance IDs/order differ')
    matrix=dist_frame.to_numpy();validate_distance_matrix(matrix,n)
    occupancy=pd.read_csv(run/'results/boundary_occupancy.csv')
    require(len(occupancy)==8*n and not occupancy.duplicated(['solution','parameter']).any(),'Boundary occupancy member/parameter rows differ')
    occupancy=occupancy.set_index(['solution','parameter'])
    records=[];bank=[];dt_expected=.025
    reference_raw=np.column_stack([np.loadtxt(baseline/(name+'.txt')) for name in RATES])
    grid=np.arange(round(horizon/dt_expected)+1)*dt_expected
    reference=np.column_stack([np.interp(grid,np.arange(len(reference_raw)),reference_raw[:,j]) for j in range(8)])
    for k,name in enumerate(expected_ids):
        meta=read_json(base/'solutions'/(name+'.json'));row=manifest.iloc[k]
        require(meta.get('solution')==name and meta.get('generation_index')==k+1 and row.generation_index==k+1,name+' member identity differs')
        require(meta.get('accepted') is True,name+' saved acceptance flag false')
        for field in ['beta','R2_GD_RK4','R2_GX_RK4','R2_GD_DOP853','R2_GX_DOP853','min_state','mass_error','bound_min','bound_max','RK4_step_error','DOP853_cumulative_error','DOP853_state_max_error','selection_distance','baseline_distance']:
            close(float(row[field]),float(meta[field]),name+' JSON/manifest '+field)
        with np.load(base/'solutions'/(name+'.npz'),allow_pickle=False) as a:
            dt=float(a['dt']);r=a['rates'].copy();x=a['states'];dop=a['DOP853_daily_states'];coordinates=a['coordinates']
            require(np.isfinite(dt) and dt==dt_expected,name+' dt differs from frozen 0.025-day rate/state grid')
            require(float(a['beta'])==beta and meta['beta']==beta,name+' fixed beta differs')
            require(r.shape==(len(grid),8) and np.isfinite(r).all(),name+' rate shape/finiteness')
            require(((r>=low)&(r<=high)).all(),name+' scientific rate boundary violation')
            require(coordinates.shape==(30,) and np.isfinite(coordinates).all(),name+' deformation coordinates differ')
            mass=state_checks(x,(len(grid),12),initial,pop.sum(),name+' RK4')
            dop_mass=state_checks(dop,(horizon+1,14),initial,pop.sum(),name+' DOP853')
            require(mass<=meta['mass_error']+1e-7,name+' retained-grid mass error exceeds recorded finer-grid maximum')
            require(float(x.min())>=meta['min_state']-1e-12,name+' saved minimum contradicts fine-grid record')
            normalized=(r-low)/(high-low)
            close(normalized.min(),meta['bound_min'],name+' bound minimum',atol=1e-12)
            close(normalized.max(),meta['bound_max'],name+' bound maximum',atol=1e-12)
            for j,parameter in enumerate(RATES):
                record=occupancy.loc[(name,parameter)]
                lower=boundary_fraction(normalized[:,j]);upper=boundary_fraction(1-normalized[:,j])
                expected=[low[j],high[j],r[:,j].min(),r[:,j].max(),.02,lower,upper,lower+upper]
                close(record[['lower_bound','upper_bound','raw_min','raw_max','threshold_fraction','near_lower_fraction','near_upper_fraction','total_boundary_fraction']].to_numpy(dtype=float),expected,name+' '+parameter+' boundary occupancy',atol=1e-10)
            daily=x[::round(1/dt)]
            fits={}
            for method,states in [('RK4',daily),('DOP853',dop)]:
                cases=np.diff(states[:,10:12],axis=0)
                for j,region in enumerate(['GD','GX']):
                    r2=float(1-np.sum((cases[:,j]-observed[:,j])**2)/np.sum((observed[:,j]-observed[:,j].mean())**2))
                    require(r2>.98,name+' recomputed '+region+' '+method+' R2 fails')
                    close(r2,meta['R2_'+region+'_'+method],name+' R2 '+region+' '+method,atol=1e-10)
                    fits[region+'_'+method]=r2
            cumulative_error=float(np.max(abs(daily[:,10:12]-dop[:,10:12])))
            close(cumulative_error,meta['DOP853_cumulative_error'],name+' cumulative solver error',atol=1e-7)
            close(float(np.max(abs(daily-dop[:,:12]))),meta['DOP853_state_max_error'],name+' state solver error',atol=1e-7)
            require(meta['RK4_step_error']<=.01 and cumulative_error<=.01 and meta['mass_error']<=.01,name+' recorded acceptance consistency fails')
        baseline_distance=functional_distance(r,reference,low,high,dt,horizon)
        close(baseline_distance,meta['baseline_distance'],name+' distance to baseline',atol=1e-10)
        recomputed=[functional_distance(r,other,low,high,dt,horizon) for other in bank]
        if k:
            close(recomputed,matrix[k,:k],name+' recomputed pairwise distances',atol=1e-10)
            for value in [row.min_distance_to_prior_new,meta['min_distance_to_prior_new'],meta['selection_distance']]:close(min(recomputed),value,name+' prior maximin distance',atol=1e-10)
        else:
            require(meta['min_distance_to_prior_new'] is None and pd.isna(row.min_distance_to_prior_new),'U01 prior distance must be absent')
            close(meta['selection_distance'],baseline_distance,'U01 initialization reference distance',atol=1e-10)
        close(row.min_distance_to_final_others,np.delete(matrix[k],k).min(),name+' final nearest-neighbour distance',atol=1e-10)
        records.append(dict(solution=name,recomputed_R2=fits,saved_RK4_grid_mass_error=mass,saved_DOP853_mass_error=dop_mass,baseline_distance=baseline_distance))
        bank.append(r)
    values=matrix[np.triu_indices(n,1)]
    return dict(n=n,members=records,pairs_recomputed=len(values),distance_min=float(values.min()),distance_median=float(np.median(values)),distance_max=float(values.max()),
                state_grid_days=dt_expected,novelty_threshold=.025,baseline_beta=beta,boundary_occupancy_rows_checked=n*8,
                scope='Saved 0.025-day RK4 states and daily DOP853 states audited; unretained 0.0125-day RK4 substeps/coarse-run estimates remain hash-protected historical evidence; no integration.')

def audit_strategy_records(run,n,protocol):
    table=pd.read_csv(run/'results/strategy_audit.csv')
    require(len(table)==n*9 and not table.duplicated(['solution','strategy']).any(),'Strategy audit row set differs')
    require(set(zip(table.solution,table.strategy))=={(i,s) for i in ids(n) for s in STRATEGIES},'Missing strategy audit row')
    numeric=table[['min_state','mass_error','RK_step_error','GD_control_days','GX_control_days','outside_mask_change']].to_numpy()
    require(np.isfinite(numeric).all(),'Nonfinite strategy audit values')
    require((table.min_state>=0).all() and (table.mass_error<.01).all() and (table.RK_step_error<.05).all(),'Recorded RK4 strategy validation fails')
    for flag in ['c_q_unchanged','beta_unchanged']:require(table[flag].eq(True).all(),flag+' false')
    require(table.outside_mask_change.eq(0).all(),'Outside risk window modified')
    for j,(region,column) in enumerate([('GD','GD_control_days'),('GX','GX_control_days')]):
        expected=[0 if strategy[j]=='N' else protocol['regions'][region]['high_days' if strategy[j]=='H' else 'low_days'] for strategy in table.strategy]
        close(table[column],expected,'Risk duration '+region,atol=1e-7)
    cross=table.strategy.isin(['HH','LL'])
    require(np.isfinite(table.loc[cross,'DOP853_cumulative_error']).all() and (table.loc[cross,'DOP853_cumulative_error']<.05).all(),'HH/LL DOP853 recorded validation fails')
    require(table.loc[~cross,'DOP853_cumulative_error'].isna().all(),'Unexpected DOP853 scope: review documentation')
    return dict(all_strategy_RK4_rows=n*9,independent_DOP853_rows=n*2,independent_DOP853_strategies=['HH','LL'],reintegrated=False)

def audit_run(project,run_name):
    project=Path(project).resolve();run=project/'analysis_runs'/run_name
    config=read_json(run/'config_resolved.json');state=read_json(run/'pipeline_state.json')
    require_completed_analysis(run,state)
    require(state['stages']['verify_integrity']['status']=='completed','Historical final verification incomplete')
    require(read_json(run/'results/integrity_check.json')['passed'] is True,'Historical final verification failed')
    source=read_json(run/'methods/source_hashes.json')
    historical_digest=hashlib.sha256(json.dumps(dict(config=config,code=source['code'],inputs=source['inputs']),sort_keys=True).encode()).hexdigest()
    require(historical_digest==state['fingerprint'],'Historical config/source/input fingerprint metadata differs')
    require_hashes(project,{p:h for p,h in source['inputs'].items() if isinstance(h,str)},'Frozen scientific source changed')
    freeze=read_json(run/'methods/ensemble_freeze.json');require_hashes(Path(freeze['base_path']),freeze['files'],'Frozen ensemble changed')
    n=config['n_solutions']
    if config['mode']=='fresh':verify_member_commits(run,n)
    scenarios=pd.read_csv(run/'results/complete_scenarios.csv')
    policy=audit_policy(scenarios,pd.read_csv(run/'results/policy_summary.csv'),n)
    rankings={};reported=scenarios[scenarios.measure=='reported']
    for name,values in [('total_reported',reported.C1+reported.C2),('gd_reported',reported.C1),('gx_reported',reported.C2)]:
        pivot=reported.assign(value=values).pivot(index='solution',columns='strategy',values='value')[STRATEGIES].to_numpy()
        matrix=(pivot[:,:,None]<pivot[:,None,:]).mean(axis=0)
        saved=pd.read_csv(run/'results'/('pairwise_ranking_'+name+'.csv'),index_col=0)
        require(saved.index.tolist()==STRATEGIES and saved.columns.tolist()==STRATEGIES,'Ranking matrix order differs')
        close(matrix,saved.to_numpy(),name+' ranking matrix',atol=1e-15,rtol=0)
        rankings[name]=dict(unanimous_pairs=int(np.count_nonzero(matrix==1)),unordered_pairs=36)
    members=audit_members(project,run,config,freeze,pd.read_csv(run/'results/ensemble_manifest.csv'))
    strategy=audit_strategy_records(run,n,read_json(project/config['risk_protocol']))
    return dict(passed=True,run=run_name,role='manuscript main ensemble' if n==100 else 'ensemble-size sensitivity',n=n,
                policy=policy,members=members,strategy=strategy,rankings=rankings,frozen_file_hashes_checked=len(freeze['files']),historical_fingerprint_metadata_verified=True,
                historical_report_sha256=sha(run/'FINAL_REPORT.md'),ODE_calls=0,
                checked_artifacts={rel:sha(run/rel) for rel in ['config_resolved.json','pipeline_state.json','methods/ensemble_freeze.json','results/ensemble_manifest.csv','results/ensemble_distances.csv','results/complete_scenarios.csv','results/policy_summary.csv','results/strategy_audit.csv','results/boundary_occupancy.csv','results/baseline_rollout.npz']},
                fingerprint_policy='Historical code fingerprints retained as provenance; current maintained code is not substituted into old run metadata.')
