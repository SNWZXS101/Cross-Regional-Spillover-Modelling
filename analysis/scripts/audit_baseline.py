from scripts.common import *
if Context().mode=='fresh':
    from scripts.core.solvers import *
import hashlib,importlib.util,ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def audit_fresh():
    r,b,cfg=read_baseline()
    obs=pd.read_excel(PROJECT/'data'/'input'/cfg['data_file'])[[cfg['gd_column'],cfg['gx_column']]].values
    if not (len(obs)==T and len(r)==T):
        raise AssertionError('audit_baseline.py: scientific/integrity check failed: len(obs)==T and len(r)==T')
    required=['config.json','training_summary.json','rollout_metrics.csv','boundary_diagnostics.json']+[k+'.txt' for k in NAMES+['beta','gamma','mu']]
    hashes={str((BASE/k).relative_to(PROJECT)):hashlib.sha256((BASE/k).read_bytes()).hexdigest() for k in required}
    for f in [PROJECT/'model'/'train_ude.py',PROJECT/'model'/'independent_ode_validation.py',PROJECT/'data'/'input'/cfg['data_file']]:hashes[str(f.relative_to(PROJECT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    summary=json.loads((BASE/'training_summary.json').read_text());boundary=json.loads((BASE/'boundary_diagnostics.json').read_text());saved=pd.read_csv(BASE/'rollout_metrics.csv')
    spec=importlib.util.spec_from_file_location('original_validator',PROJECT/'model'/'independent_ode_validation.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    # Extract the actual current training source's independent scalar RHS without executing training.
    tree=ast.parse((PROJECT/'model'/'train_ude.py').read_text(encoding='utf-8-sig'))
    defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['interp_daily','numpy_rhs']]
    scope={'np':np,'math':__import__('math')};exec(compile(ast.Module(body=defs,type_ignores=[]),'training_rhs','exec'),scope)
    rr={k:r[:,j] for j,k in enumerate(NAMES)}
    for t in [0.,.2,10.5,125.4,235.9]:
        x=X0.copy()+np.arange(12)*.7
        if not (np.allclose(rhs(x,interp(r,t,1.),b),v.rhs(t,x,rr,b,G,MU,*N),rtol=1e-12,atol=1e-10)):
            raise AssertionError('audit_baseline.py: scientific/integrity check failed: np.allclose(rhs(x,interp(r,t,1.),b),v.rhs(t,x,rr,b,G,MU,*N),rtol=1e-12,atol=1e-10)')
        if not (np.allclose(scope['numpy_rhs'](t,x,rr,b,G,MU,*N),v.rhs(t,x,rr,b,G,MU,*N),rtol=1e-12,atol=1e-10)):
            raise AssertionError("audit_baseline.py: scientific/integrity check failed: np.allclose(scope['numpy_rhs'](t,x,rr,b,G,MU,*N),v.rhs(t,x,rr,b,G,MU,*N),rtol=1e-12,atol=1e-10)")
    xx=rollout(r,1.,b,T,80);coarse=rollout(r,1.,b,T,40);dd=xx[::80]
    independent=dop853(r,1.,b,('N','N'))
    original,warnings=v.rk4(X0,rr,b,G,MU,*N,T,40)
    rows=[]
    for name,x in [('RK4_h0.0125',dd),('DOP853',independent),('original_validator_RK4_h0.025',original)]:
        for j,reg in enumerate(['GD','GX']):rows.append(dict(method=name,region=reg,**metrics(obs[:,j],np.diff(x[:,10+j]))))
    pd.DataFrame(rows).to_csv(ROOT/'results/baseline_metrics.csv',index=False)
    cv=(r-LOW)/(HIGH-LOW);mass=float(abs(xx[:,:10].sum(axis=1)-N.sum()).max())
    err=float(abs(dd[:,10:12]-independent[:,10:12]).max())
    audit=dict(passed=bool(min(z['R2'] for z in rows)>.98 and xx.min()>=0 and cv.min()>=0 and cv.max()<=1 and BETA_BOUNDS[0]<=b<=BETA_BOUNDS[1] and mass<.01 and err<.01),baseline=str(BASE),beta=b,gamma=G,mu=MU,bounds={k:cfg[k+'_bounds'] for k in NAMES+['beta']},initial_state=X0.tolist(),horizon=T,metrics=rows,min_state=float(xx.min()),total_population_max_error=mass,total_population_relative_error=mass/N.sum(),DOP853_cumulative_max_error=err,RK4_step_error=float(abs(dd-coarse[::40]).max()),original_scalar_validator_max_error=float(abs(dd-original).max()),source_RHS_equivalence=True,sha256=hashes,saved_training_summary=summary,saved_boundary_diagnostics=boundary,saved_rollout_metrics=saved.to_dict('records'),rates={k:r[:,j].tolist() for j,k in enumerate(NAMES)})
    (ROOT/'results/baseline_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    np.savez_compressed(ROOT/'results/baseline_rollout.npz',time=np.arange(T+1),rates=r,states=dd,DOP853=independent,observed=obs)
    fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for j,ax in enumerate(axs):
        ax.plot(obs[:,j],'k.',ms=3,label='Observed');ax.plot(np.diff(dd[:,10+j]),label='RK4 h=0.0125');ax.plot(np.diff(independent[:,10+j]),'--',label='Independent DOP853');ax.set_ylabel(['GD','GX'][j]+' reported cases');ax.legend();ax.grid(alpha=.2)
    axs[-1].set_xlabel('Day');fig.tight_layout()
    for name in ['baseline_independent_rollout.png','baseline_independent_fit.png']:fig.savefig(ROOT/'figures'/name,dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(4,2,figsize=(12,11),sharex=True)
    for j,ax in enumerate(axs.flat):ax.plot(r[:,j]);ax.set_title(NAMES[j]);ax.set_ylim(LOW[j],HIGH[j]);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'figures/baseline_parameter_profiles.png',dpi=180);plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False));print('BASELINE PASSED:',audit['passed'],'mass error',mass,'DOP853 error',err)
    if not (audit['passed']):
        raise AssertionError("audit_baseline.py: scientific/integrity check failed: audit['passed']")


def main():
    ctx=Context()
    if ctx.mode=='fresh':
        audit_fresh();return
    check_frozen_source(ctx)
    mapping={'results/new_baseline_audit.json':'results/baseline_audit.json','results/new_baseline_rollout_metrics.csv':'results/baseline_metrics.csv','results/new_baseline_rollout.npz':'results/baseline_rollout.npz','figures/new_baseline_independent_rollout.png':'figures/baseline_independent_rollout.png','figures/new_baseline_independent_fit.png':'figures/baseline_independent_fit.png','figures/new_baseline_parameter_profiles.png':'figures/baseline_parameter_profiles.png'}
    provenance=[copy_existing(ctx,a,b) for a,b in mapping.items()]
    if not read_json(ctx.results/'baseline_audit.json')['passed']:raise RuntimeError('Archived baseline did not pass')
    atomic_json(ctx.root/'methods/baseline_reuse.json',dict(mode='verified archive reuse; no ODE integration',files=provenance))
    print('Reused baseline audit byte-for-byte; ODE calls=0')

if __name__=='__main__':main()
