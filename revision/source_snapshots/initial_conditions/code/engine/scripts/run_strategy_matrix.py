from scripts.common import *
if Context().mode=='fresh':
    from scripts.core.solvers import *
from concurrent.futures import ProcessPoolExecutor
import hashlib

def evaluate(name):
    target=ROOT/'results/scenarios';target.mkdir(exist_ok=True)
    marker=ROOT/'checkpoints'/(name+'_strategies.json')
    if marker.exists():
        for rel,h in read_json(marker)['files'].items():
            if sha(ROOT/rel)!=h:raise RuntimeError('Scenario checkpoint checksum mismatch: '+rel)
        return name+' already complete (verified)'
    a=np.load(ROOT/'solutions'/(name+'.npz'));r=a['rates'];b=float(a['beta']);dt=float(a['dt'])
    rate_hash=hashlib.sha256(r.tobytes()).hexdigest()
    tt=grid(.025);di=np.searchsorted(tt,np.arange(T+1));rows=[];aud=[];traj={};full={}
    for s in STRATEGIES:
        tag=''.join(s);fac=factors(tt,s);fine,mn,mass=simulate(r,dt,b,tt,fac,.00625);coarse,_,_=simulate(r,dt,b,tt,fac,.0125,False)
        step_error=float(abs(fine[-1,10:]-coarse[-1,10:]).max())
        if not (mn>=0 and mass<.01 and step_error<.05):
            raise AssertionError((name,tag,mn,mass,step_error))
        dd=fine[di];traj[tag]=dd;full[tag]=fine
        if not np.array_equal(full[tag][di],dd):raise RuntimeError('Fine-grid/daily trajectory mismatch')
        cross_error=np.nan
        if tag in ['HH','LL']:
            independent=dop853_strategy(r,dt,b,s);cross_error=float(abs(independent[:,10:]-dd[:,10:]).max());
            if not (cross_error<.05):
                raise AssertionError((name,tag,cross_error))
        selected=fac[:,4:6]<1;dur=np.diff(tt)@selected
        if not (np.array_equal(fac[:,:4],np.ones_like(fac[:,:4]))):
            raise AssertionError('run_strategy_matrix.py: scientific/integrity check failed: np.array_equal(fac[:,:4],np.ones_like(fac[:,:4]))')
        for j in range(2):
            if not (abs(dur[j]-(0 if s[j]=='N' else 118))<1e-7):
                raise AssertionError("run_strategy_matrix.py: scientific/integrity check failed: abs(dur[j]-(0 if s[j]=='N' else 118))<1e-7")
            if not (np.all(fac[~selected[:,j],4+j]==1) and np.all(fac[~selected[:,j],6+j]==1)):
                raise AssertionError('run_strategy_matrix.py: scientific/integrity check failed: np.all(fac[~selected[:,j],4+j]==1) and np.all(fac[~selected[:,j],6+j]==1)')
        aud.append(dict(solution=name,strategy=tag,GD_control_days=float(dur[0]),GX_control_days=float(dur[1]),c_q_unchanged=True,beta_unchanged=True,outside_mask_change=0,min_state=mn,mass_error=mass,RK_step_error=step_error,DOP853_cumulative_error=cross_error))
        for measure,sl in [('reported',slice(10,12)),('all_infections',slice(12,14))]:
            C=dd[-1,sl];avoided=traj['NN'][-1,sl]-C
            rows.append(dict(solution=name,strategy=tag,GD_strategy=s[0],GX_strategy=s[1],measure=measure,C1=float(C[0]),C2=float(C[1]),A1=float(avoided[0]),A2=float(avoided[1]),Atotal=float(avoided.sum()),classification=classify(avoided,s),win_win=bool((avoided>.01).all()),beta=b))
    if not (hashlib.sha256(r.tobytes()).hexdigest()==rate_hash):
        raise AssertionError('run_strategy_matrix.py: scientific/integrity check failed: hashlib.sha256(r.tobytes()).hexdigest()==rate_hash')
    pd.DataFrame(rows).to_csv(target/(name+'_scenarios.csv'),index=False)
    np.savez_compressed(target/(name+'_daily_trajectories.npz'),time=np.arange(T+1),**traj)
    np.savez_compressed(target/(name+'_fine_trajectories.npz'),time=tt,**full)
    pd.DataFrame(aud).to_csv(target/(name+'_audit.csv'),index=False)
    atomic_json(marker,dict(files={str(p.relative_to(ROOT)):sha(p) for p in [target/(name+'_scenarios.csv'),target/(name+'_daily_trajectories.npz'),target/(name+'_fine_trajectories.npz'),target/(name+'_audit.csv')]}))
    return name+' complete: all 9 scenarios, fine RK4; HH/LL independent DOP853'

def strategy_fresh():
    ctx=Context();verify_ensemble(ctx)
    m=pd.read_csv(ROOT/'results/ensemble_manifest.csv')
    if m.solution.tolist()!=ids(ctx.n):raise RuntimeError('Wrong frozen member set')
    with ProcessPoolExecutor(max_workers=Context().config['workers']) as pool:
        for msg in pool.map(evaluate,m.solution):print(msg,flush=True)
    target=ROOT/'results/scenarios'
    rows=pd.concat([pd.read_csv(target/(name+'_scenarios.csv')) for name in m.solution],ignore_index=True)
    if not (len(rows)==Context().n*9*2):
        raise AssertionError('run_strategy_matrix.py: scientific/integrity check failed: len(rows)==Context().n*9*2')
    rows.to_csv(ROOT/'results/complete_scenarios.csv',index=False)
    pd.concat([pd.read_csv(target/(name+'_audit.csv')) for name in m.solution],ignore_index=True).to_csv(ROOT/'results/strategy_audit.csv',index=False)
    # The same nine-strategy mask/multiplier sequence applies to every member.
    tt=grid(.025);controls=pd.DataFrame(dict(start=tt[:-1],end=tt[1:]))
    for s in STRATEGIES:
        fac=factors(tt,s)
        for j,name in enumerate(NAMES[4:]):controls[''.join(s)+'_'+name+'_multiplier']=fac[:,4+j]
    controls.to_csv(ROOT/'methods/explicit_strategy_multipliers.csv',index=False)


def main():
    ctx=Context();verify_ensemble(ctx)
    if ctx.mode=='fresh':strategy_fresh();return
    provenance=[copy_existing(ctx,'results/uniform_30_complete_scenarios_v2.csv','results/complete_scenarios.csv'),copy_existing(ctx,'results/uniform_30_strategy_audit_v2.csv','results/strategy_audit.csv'),copy_existing(ctx,'results/uniform_30_policy_summary_v2.csv','results/archived_policy_summary.csv')]
    validate_scenarios(pd.read_csv(ctx.results/'complete_scenarios.csv'),ctx.n)
    atomic_json(ctx.root/'methods/strategy_reuse.json',dict(mode='byte-identical archived CSV; no scenario integration',files=provenance))
    print('Reused 270 existing scenarios; ODE calls=0')

if __name__=='__main__':main()
